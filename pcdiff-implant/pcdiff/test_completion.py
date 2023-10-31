import argparse
import torch.nn as nn
import torch.utils.data

from torch.distributions import Normal
from model.pvcnn_completion import PVCNN2Base
from utils.file_utils import *
from tqdm import tqdm
from datasets.skullbreak_data import SkullBreakDataset
from datasets.skullfix_data import SkullFixDataset
'''
----- Models -----
'''

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.ones_like(cdf_min) * 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < 0.001, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(
        torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        #assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])

        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step) """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)

        assert noise.shape == x_start.shape

        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start + self._extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0) """
        assert x_start.shape == x_t.shape
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)[:, :, self.sv_points:]

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = self.posterior_variance.to(data.device), \
                                                 self.posterior_log_variance_clipped.to(data.device)

            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(model_output)

        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data[:, :, self.sv_points:], t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data[:, :, self.sv_points:], t=t)

        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon

        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape

        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - pred_xstart) / \
               self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)

    ''' 
    ----- DDPM sampling ----- 
    '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """ Sample from the model """
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, data=data, t=t,
                                                                 clip_denoised=clip_denoised, return_pred_xstart=False)

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return sample

    def p_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True,
                      keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """

        assert isinstance(shape, (tuple, list))
        noise = noise_fn(size=shape, dtype=torch.float, device=device)

        img_t = torch.cat([partial_x, noise], dim=-1)

        for t in tqdm(reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))), total=1000):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t[:, :, self.sv_points:].shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, noise_fn=torch.randn, clip_denoised=True,
                                 keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps = self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0, total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)
            if t % freq == 0 or t == total_steps - 1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    '''
    ----- DDIM sampling -----
    '''
    def ddim_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=True, return_pred_xstart=True, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        model_mean, _, _, x_start = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                         return_pred_xstart=return_pred_xstart)

        eps = self._predict_eps_from_xstart(data[:, :, self.sv_points:], t, x_start)

        alpha_bar = self._extract(self.alphas_cumprod.to(data.device), t, data.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev.to(data.device), t, data.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)
        mean_pred = (x_start * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(data.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return sample

    def ddim_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True,
                         sampling_steps=1000):

        assert isinstance(shape, (tuple, list))
        noise = noise_fn(size=shape, dtype=torch.float, device=device)

        img_t = torch.cat([partial_x, noise], dim=-1)

        ts = np.linspace(0, 999, sampling_steps).round().astype('int')
        ts = np.unique(ts)[::-1]

        for t in tqdm(ts):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.ddim_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                     clip_denoised=clip_denoised, return_pred_xstart=True)

        assert img_t[:, :, self.sv_points:].shape == shape
        return img_t

    '''
    ----- Losses -----
    '''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start[:, :, self.sv_points:], x_t=data_t[:, :, self.sv_points:], t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data_t, t=t,
                                                                              clip_denoised=clip_denoised,
                                                                              return_pred_xstart=True)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(model_mean.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """ Training loss calculation """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start[:, :, self.sv_points:].shape, dtype=data_start.dtype,
                                device=data_start.device)

        data_t = self.q_sample(x_start=data_start[:, :, self.sv_points:], t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(torch.cat([data_start[:, :, :self.sv_points], data_t], dim=-1), t)[:, :, self.sv_points:]
            losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start.shape))))

        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t,
                                        clip_denoised=False, return_pred_xstart=False)

        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])

        return losses

    '''
    ----- Debug -----
    '''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T - 1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=torch.tensor([0.]).to(qt_mean),
                                 logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_ = torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):
                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                data_t = torch.cat(
                    [x_start[:, :, :self.sv_points], self.q_sample(x_start=x_start[:, :, self.sv_points:], t=t_b)],
                    dim=-1)
                new_vals_b, pred_xstart = self._vb_terms_bpd(denoise_fn, data_start=x_start, data_t=data_t, t=t_b,
                                                             clip_denoised=clip_denoised, return_pred_xstart=True)

                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start[:, :, self.sv_points:].shape
                new_mse_b = ((pred_xstart - x_start[:, :, self.sv_points:]) ** 2).mean(
                    dim=list(range(1, len(pred_xstart.shape))))
                assert new_vals_b.shape == new_mse_b.shape == torch.Size([B])

                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None] == torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start[:, :, self.sv_points:])
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
            assert vals_bt_.shape == mse_bt_.shape == torch.Size(
                [B, T]) and total_bpd_b.shape == prior_bpd_b.shape == torch.Size([B])
            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class PVCNN2(PVCNN2Base):
    num_n = 128

    # Define set abstraction layers
    sa_blocks = [((32, 2, 32), (10240, 0.1, num_n, (32, 64))),
                 ((64, 3, 16), (2560, 0.2, num_n, (64, 128))),
                 ((128, 3, 8), (640, 0.4, num_n, (128, 256))),
                 (None, (160, 0.8, num_n, (256, 256, 512))),
                 ]

    # Define feature propagation layers
    fp_blocks = [((256, 256), (256, 3, 8)),
                 ((256, 256), (256, 3, 8)),
                 ((256, 128), (128, 2, 16)),
                 ((128, 128, 64), (64, 2, 32)),
                 ]

    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str,
                 width_mult: float, vox_res_mult: float):
        super(Model, self).__init__()

        # Create diffusion (DDPM)
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type,
                                           sv_points=(args.num_points-args.num_nn))

        # Create diffusion (DDIM)
        if args.sampling_method == 'ddim':
            timesteps = np.linspace(0, args.time_num-1, args.sampling_steps).round().astype('int')
            timesteps = np.unique(timesteps)[::-1]
            timesteps = set(timesteps)

            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(self.diffusion.alphas_cumprod):
                if i in timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                else:
                    new_betas.append(0.0)
            new_betas = np.array(new_betas)
            self.diffusion = GaussianDiffusion(new_betas, loss_type, model_mean_type, model_var_type,
                                               sv_points=(args.num_points-args.num_nn))

        # Create point-voxel-cnn network
        self.model = PVCNN2(num_classes=args.nc, sv_points=(args.num_points-args.num_nn), embed_dim=args.embed_dim,
                            use_att=args.attention, dropout=args.dropout, extra_feature_channels=0,
                            width_multiplier=width_mult, voxel_resolution_multiplier=vox_res_mult)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {'total_bpd_b': total_bpd_b,
                'terms_bpd': vals_bt,
                'prior_bpd_b': prior_bpd_b,
                'mse_bt': mse_bt}

    def _denoise(self, data, t):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, partial_x, shape, device, noise_fn=torch.randn, clip_denoised=True, keep_running=False,
                    sampling_method='ddpm', sampling_steps=1000):
        if sampling_method == 'ddpm':
            return self.diffusion.p_sample_loop(partial_x, self._denoise, shape=shape, device=device, noise_fn=noise_fn,
                                                clip_denoised=clip_denoised, keep_running=keep_running)
        if sampling_method == 'ddim':
            return self.diffusion.ddim_sample_loop(partial_x, self._denoise, shape=shape, device=device,
                                                   noise_fn=noise_fn, clip_denoised=clip_denoised,
                                                   sampling_steps=sampling_steps)
        else:
            raise NotImplementedError("Not implemented. Use 'ddpm' or 'ddim'.")


    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    else:
        raise NotImplementedError(schedule_type)

    return betas


#############################################################################
def get_dataset(path, num_points, num_nn, dataset):
    if dataset == 'SkullBreak':
        te_dataset = SkullBreakDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode='shape_bbox',
                                       eval=True)
    else:
        te_dataset = SkullFixDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode='shape_bbox',
                                     eval=True)
    return te_dataset


def evaluate_recon_mvr(opt, model, save_dir):
    test_dataset = get_dataset(opt.path, opt.num_points, opt.num_nn, opt.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs, shuffle=False,
                                                  num_workers=int(opt.workers), drop_last=False)

    model.eval()

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Reconstructing Samples'):
        pc = data['train_points'].transpose(1, 2).to("cuda:" + str(opt.gpu))
        name = (data['name'][0].split('/')[-2] + data['name'][0].split('/')[-1].split('.')[0])

        ensemble_num = opt.num_ens

        pc = pc.repeat(ensemble_num, 1, 1)
        noise_shape = torch.Size([ensemble_num, 3, opt.num_nn])

        with torch.no_grad():
            sample = model.gen_samples(pc, noise_shape, pc.device, clip_denoised=False,
                                       sampling_method=opt.sampling_method, sampling_steps=opt.sampling_steps)
            sample = sample.detach().cpu()

        sample_np = np.asarray(sample)
        sample_np = sample_np.transpose(0, 2, 1)

        if not (os.path.exists(os.path.join(save_dir, name))):
            os.mkdir(os.path.join(save_dir, name))

        np.save(os.path.join(save_dir, name, 'input.npy'), np.asarray(data['train_points']))  # save the input points
        np.save(os.path.join(save_dir, name, 'sample.npy'), sample_np)  # save the sampled point clouds (implants)
        np.save(os.path.join(save_dir, name, 'shift.npy'), np.asarray(data['shift']))  # save shift
        np.save(os.path.join(save_dir, name, 'scale.npy'), np.asarray(data['scale']))  # save scale
    return


def main(opt):
    output_dir = opt.eval_path
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger = setup_logging(output_dir)
    outf_syn, = setup_output_subdirs(output_dir, 'syn')

    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type, width_mult=opt.width_mult,
                  vox_res_mult=opt.vox_res_mult)

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    torch.cuda.set_device(opt.gpu)
    model = model.cuda(opt.gpu)

    if opt.distr_train:
        model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():
        logger.info("Perform sampling with:%s" % opt.model)

        resumed_param = torch.load(opt.model, map_location=("cuda:" + str(opt.gpu)))
        model.load_state_dict(resumed_param['model_state'])

        if opt.eval_recon_mvr:
            # Evaluate generation
            evaluate_recon_mvr(opt, model, outf_syn)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="set the path to the dataset here")
    parser.add_argument('--dataset', type=str, help="specify the used dataset (SkullBreak or SkullFix)")
    parser.add_argument('--model', default='', required=True, help="path to model to sample from")
    parser.add_argument('--num_ens', type=int, default=1, help='number of samples for ensembling')
    parser.add_argument('--sampling_method', type=str, default='ddpm', help='ddpm or ddim')
    parser.add_argument('--sampling_steps', type=int, default=1000)

    parser.add_argument('--bs', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=24, help='workers')
    parser.add_argument('--niter', type=int, default=100000, help='number of epochs to train for')  # not used

    parser.add_argument('--eval_recon_mvr', type=eval, default=True)
    parser.add_argument('--eval_saved', type=eval, default=True)

    parser.add_argument('--nc', type=int, default=3, help="dimension of one point (usually 3 for x, y,z)")
    parser.add_argument('--num_points', type=int, default=30720, help="number of points the point cloud should contain")
    parser.add_argument('--num_nn', type=int, default=3072, help="number of points that represent the implant")

    '''model'''
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', type=str, default='linear')
    parser.add_argument('--time_num', type=int, default=1000)

    # params
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--model_mean_type', type=str, default='eps')
    parser.add_argument('--model_var_type', type=str, default='fixedsmall')
    parser.add_argument('--vox_res_mult', type=float, default=1.0)
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--distr_train', type=eval, default=False)

    '''eval'''
    parser.add_argument('--eval_path', default='', required=True, help='set manual path to save the results')
    parser.add_argument('--manualSeed', default=48, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use. None means using all available GPUs.')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        opt.cuda = True
    else:
        opt.cuda = False

    return opt


if __name__ == '__main__':
    opt = parse_args()

    main(opt)
