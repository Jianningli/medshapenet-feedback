import argparse
import diplib as dip
import nrrd
import numpy as np
import os
import torch
import yaml
from eval_metrics import bdc, dc, hd95
from src import config
from src.model import Encode2Points
from src.data.core import SkullEval
from src.utils import load_config, load_model_manual, readCT, crop, padding, reverse_padding, reverse_crop, \
    re_sample_shape, filter_voxels_within_radius
from tqdm import tqdm
from scipy import ndimage

np.set_printoptions(precision=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file you want to use.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Set a random seed (default: 1)')
    parser.add_argument('--iter', type=int, metavar='S', help='The training iteration to be evaluated.')

    args = parser.parse_args()

    cfg = load_config(args.config, 'voxelization/configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dev = "cuda:" + str(cfg['train']['gpu'])
    device = torch.device(dev if use_cuda else "cpu")
    vis_n_outputs = cfg['generation']['vis_n_outputs']

    if vis_n_outputs is None:
        vis_n_outputs = -1

    # Shorthands
    out_dir = cfg['train']['out_dir']
    if not out_dir:
        os.makedirs(out_dir)

    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    dataset = SkullEval(cfg['data']['path'])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    model = Encode2Points(cfg).to(device)

    # load model
    print('\n---------- Load model----------')
    print("Load best model from: " + cfg['test']['model_file'] + '\n')
    state_dict = torch.load(cfg['test']['model_file'], map_location=device)
    load_model_manual(state_dict['state_dict'], model)

    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Generate
    model.eval()

    print('\n---------- Voxelizing point clouds ----------')
    for it, data in enumerate(test_loader):
        print("Sampling step [" + str(it+1) + "/" + str(len(test_loader)) + "] ...")
        name = data['name'][0]

        # ----- Get defective skull -----
        # Simply load from SkullBreak dataset (no preprocessing)
        if cfg['data']['dset'] == 'SkullBreak':
            defective_skull, header = nrrd.read(os.path.join(name.split('/results')[0], 'defective_skull',
                                                name.split('syn/')[1][:-8], name.split('_surf')[0][-3:] + '.nrrd'))

        # Load from SkullFix dataset with resampling, cropping and zero padding
        if cfg['data']['dset'] == 'SkullFix':
            defective_skull = readCT(os.path.join(name.split('/results')[0], 'defective_skull',
                                                  name.split('_surf')[0][-3:] + '.nrrd'))
            defective_skull, idx_x, idx_y, idx_z, shape = crop(defective_skull)
            defective_skull, dim_x, dim_y, dim_z = padding(defective_skull)
            defective_skull = defective_skull.astype(np.float32)

        inputs = data['inputs'][0, :, :, :]  # Input point cloud
        completes = np.zeros((512, 512, 512))

        # Perform generation with ensembling method (requires to also run point cloud diffusion model with ensembling flag)
        if cfg['generation']['num_ensemble'] >= 2:
            for pc in tqdm(range(cfg['generation']['num_ensemble']), total=cfg['generation']['num_ensemble']):
                # Generate a sample
                vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs[pc, :, :].unsqueeze(dim=0))

                # Generate binary segmentation mask
                psr_grid = psr_grid.detach().cpu().numpy()
                psr_grid = psr_grid[0, :, :, :]
                out = np.zeros((512, 512, 512))
                out[psr_grid <= 0] = 1
                out = ndimage.binary_dilation(out)

                completes += out  # Add to implant ensemble

                if cfg['generation']['save_ensemble_implants']:
                    # Postprocess the single implants and save the implants of the ensemble
                    out = out - defective_skull
                    out = dip.MedianFilter(out, dip.Kernel(shape='rectangular', param=(3, 3, 3)))
                    out.Convert('BIN')
                    out = dip.Closing(out, dip.SE((3, 3, 3)))
                    out = dip.Closing(out, dip.SE((3, 3, 3)))
                    out = dip.FillHoles(out)
                    out = dip.Label(out, mode='largest')
                    out = np.asarray(out, dtype=np.float32)

                    if cfg['data']['dset'] == 'SkullFix':
                        util, header = nrrd.read(os.path.join(name.split('/results')[0], 'defective_skull',
                                                              name.split('_surf')[0][-3:] + '.nrrd'))
                        out = reverse_padding(out, dim_x, dim_y, dim_z)
                        out = reverse_crop(out, idx_x, idx_y, idx_z, shape)

                        new_shape = np.asarray(util.shape)
                        out_re, _ = re_sample_shape(out, [0.45, 0.45, 0.45], new_shape)
                        im = np.zeros(out_re.shape)
                        im[out_re > 0.5] = 1
                        nrrd.write(str(name + '/impl_' + str(pc) + '.nrrd'), im, header)

                    if cfg['data']['dset'] == 'SkullBreak':
                        nrrd.write(str(name + '/impl_' + str(pc) + '.nrrd'), out, header)

        # Performs generation without ensembling
        else:
            # Generate a sample
            vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs[0, :, :].unsqueeze(dim=0))

            # Generate binary segmentation mask
            psr_grid = psr_grid.detach().cpu().numpy()
            psr_grid = psr_grid[0, :, :, :]
            out = np.zeros((512, 512, 512))
            out[psr_grid <= 0] = 1
            out = ndimage.binary_dilation(out)

            completes += out  # Add to implant ensemble

        # Compute mean implant
        mean_complete = np.zeros((512, 512, 512))
        mean_complete[completes >= np.ceil(cfg['generation']['num_ensemble']/2)] = 1

        mean_implant = mean_complete - defective_skull
        mean_implant = filter_voxels_within_radius(inputs[0, 30720-3072:, :], mean_implant)
        mean_implant = mean_implant.astype(bool)
        mean_implant = dip.Opening(mean_implant, dip.SE((3, 3, 3)))
        #mean_implant = dip.Opening(mean_implant, dip.SE((3, 3, 3)))
        mean_implant = dip.Label(mean_implant, mode="largest")
        mean_implant = dip.MedianFilter(mean_implant, dip.Kernel(shape='rectangular', param=(3, 3, 3)))
        mean_implant.Convert('BIN')
        mean_implant = dip.Closing(mean_implant, dip.SE((3, 3, 3)))
        #mean_implant = dip.Closing(mean_implant, dip.SE((3, 3, 3)))
        mean_implant = dip.FillHoles(mean_implant)

        if cfg['data']['dset'] == 'SkullBreak':
            mean_implant = dip.Label(mean_implant, mode='largest')
            mean_implant = np.asarray(mean_implant, dtype=np.float32)
            gt_implant, _ = nrrd.read(os.path.join(name.split('/results')[0], 'implant',
                                                   name.split('syn/')[1][:-8], name.split('_surf')[0][-3:] + '.nrrd'))
            defective_skull, header = nrrd.read(os.path.join(name.split('/results')[0], 'defective_skull',
                                                             name.split('syn/')[1][:-8], name.split('_surf')[0][-3:] + '.nrrd'))

        if cfg['data']['dset'] == 'SkullFix':
            mean_implant = dip.Label(mean_implant, mode='largest')
            mean_implant = np.asarray(mean_implant, dtype=np.float32)
            gt_implant, _ = nrrd.read(os.path.join(name.split('/results')[0], 'implant',
                                                   name.split('_surf')[0][-3:] + '.nrrd'))
            defective_skull, header = nrrd.read(os.path.join(name.split('/results')[0], 'defective_skull',
                                                        name.split('_surf')[0][-3:] + '.nrrd'))

        new_shape = np.asarray(defective_skull.shape)
        spacing = np.asarray([header['space directions'][0, 0],
                              header['space directions'][1, 1],
                              header['space directions'][2, 2]])

        # Rescale to original image size and voxel spacing
        if cfg['data']['dset'] == 'SkullFix':
            mean_implant = reverse_padding(mean_implant, dim_x, dim_y, dim_z)
            mean_implant = reverse_crop(mean_implant, idx_x, idx_y, idx_z, shape)
            mean_implant_re, _ = re_sample_shape(mean_implant, [0.45, 0.45, 0.45], new_shape)
            mean_implant = np.zeros(mean_implant_re.shape)
            mean_implant[mean_implant_re > 0.5] = 1
            mean_implant = mean_implant.astype(bool)
            mean_implant = dip.Label(mean_implant, mode='largest')
            mean_implant = np.asarray(mean_implant, dtype=np.float32)

        # Finally save the mean implant
        nrrd.write(str(name + '/mean_impl.nrrd'), mean_implant, header)

        # Compute the evaluation metrics
        if cfg['generation']['compute_eval_metrics']:
            eval_mets = dict()

            # Generate evaluation metrics
            print("Compute eval metrics for: " + name)
            print("Voxelspacing: " + str(spacing))

            dice = dc(mean_implant, gt_implant)
            eval_mets['dice'] = dice
            print("Dice score: " + str(dice))

            bdice = bdc(mean_implant, gt_implant, defective_skull, voxelspacing=spacing)
            eval_mets['bdice'] = bdice
            print("Boundary dice (10mm): " + str(bdice))

            haussdorf95 = hd95(mean_implant, gt_implant, voxelspacing=spacing)
            eval_mets['haussdorf95'] = float(haussdorf95)
            print("95 percentile Haussdorf distance: " + str(haussdorf95) + '\n')

            # Write scores to .yaml file
            with open(name + '/eval_metrics.yaml', 'w') as file:
                documents = yaml.dump(eval_mets, file)


if __name__ == '__main__':
    main()
