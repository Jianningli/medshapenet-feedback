import os
import torch
import torch.optim as optim
import numpy as np
import shutil
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from src import config
from src.data import collate_remove_none, collate_stack_together, worker_init_fn, SkullDataset
from src.training import Trainer
from src.model import Encode2Points
from src.utils import load_config, initialize_logger, AverageMeter, load_model_manual

np.set_printoptions(precision=4)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Set a random seed (default: 1)')
    
    args = parser.parse_args()
    cfg = load_config(args.config, 'configs/default.yaml')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dev = "cuda:" + str(cfg['train']['gpu'])
    device = torch.device(dev if use_cuda else "cpu")
    input_type = cfg['data']['input_type']  # point cloud
    batch_size = cfg['train']['batch_size']  # 1
    model_selection_metric = cfg['train']['model_selection_metric']  # loss

    # PYTORCH VERSION > 1.0.0
    assert(float(torch.__version__.split('.')[-3]) > 0)

    # boiler-plate
    if cfg['train']['timestamp']:
        cfg['train']['out_dir'] += '_' + time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = initialize_logger(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    shutil.copyfile(args.config, os.path.join(cfg['train']['out_dir'], 'config.yaml'))

    logger.info("using GPU: " + torch.cuda.get_device_name(0))

    # TensorboardX writer
    tblogdir = os.path.join(cfg['train']['out_dir'], "tensorboard_log")
    if not os.path.exists(tblogdir):
        os.makedirs(tblogdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tblogdir)

    inputs = None

    # Dataloader for training set
    train_dataset = SkullDataset(cfg['data']['train_path'], 'training', noise_stddev=cfg['data']['pointcloud_noise'])

    # Dataloader for validation set
    val_dataset = SkullDataset(cfg['data']['eval_path'], 'eval', noise_stddev=cfg['data']['pointcloud_noise'])
    vis_dataset = SkullDataset(cfg['data']['eval_path'], 'viz', noise_stddev=cfg['data']['pointcloud_noise'])

    collate_fn = collate_remove_none

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=cfg['train']['n_workers'],
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=cfg['train']['n_workers'],
                                             shuffle=True,
                                             collate_fn=collate_fn,
                                             worker_init_fn=worker_init_fn)

    vis_loader = torch.utils.data.DataLoader(vis_dataset,
                                             batch_size=1,
                                             num_workers=cfg['train']['n_workers_val'],
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             worker_init_fn=worker_init_fn)
    
    if torch.cuda.device_count() > 1:
        model = Encode2Points(cfg).to(device)
        #model = torch.nn.DataParallel(Encode2Points(cfg)).to(device)
    else:
        model = Encode2Points(cfg).to(device)

    n_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Number of parameters: %d'% n_parameter)
    # load model
    try:
        # load model
        state_dict = torch.load(os.path.join(cfg['train']['out_dir'], 'model.pt'))
        load_model_manual(state_dict['state_dict'], model)
            
        out = "Load model from iteration %d" % state_dict.get('it', 0)
        logger.info(out)
        # load point cloud
    except:
        state_dict = dict()
    
    metric_val_best = state_dict.get('loss_val_best', np.inf)

    logger.info('Current best validation metric (%s): %.8f' % (model_selection_metric, metric_val_best))

    LR = float(cfg['train']['lr'])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = state_dict.get('epoch', -1)
    it = state_dict.get('it', -1)

    trainer = Trainer(cfg, optimizer, device=device)
    runtime = {}
    runtime['all'] = AverageMeter()
    
    # Training loop
    for epoch in range(start_epoch+1, cfg['train']['total_epochs']+1):
        for batch in train_loader:
            it += 1
            
            start = time.time()

            # perform one training step
            loss, loss_each = trainer.train_step(inputs, batch, model)

            # measure elapsed time
            end = time.time()
            runtime['all'].update(end - start)

            # Print to logger
            if it % cfg['train']['print_every'] == 0:
                log_text = ('[Epoch %02d] it=%d, loss=%.4f') %(epoch, it, loss)
                writer.add_scalar('train/loss', loss, it)
                if loss_each is not None:
                    for k, l in loss_each.items():
                        if l.item() != 0.:
                            log_text += (' loss_%s=%.4f') % (k, l.item())
                        writer.add_scalar('train/%s' % k, l, it)
                
                log_text += (' time=%.3f / %.2f') % (runtime['all'].val, runtime['all'].sum)
                logger.info(log_text)

            # Visualize some results
            if (it > 0) & (it % cfg['train']['visualize_every'] == 0):
                for i, batch_vis in enumerate(vis_loader):
                    trainer.save(model, batch_vis, it, i)
                    if i >= 4:
                        break
                logger.info('Saved mesh and pointcloud')

        # Run validation
        if epoch > 0 and (epoch % cfg['train']['validate_every']) == 0:
            eval_dict = trainer.evaluate(val_loader, model)
            metric_val = eval_dict[model_selection_metric]
            logger.info('Validation metric (%s): %.4f' % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                writer.add_scalar('val/%s' % k, v, it)

            if -(metric_val - metric_val_best) >= 0:
                metric_val_best = metric_val
                logger.info('New best model (loss %.4f), epoch %d' % (metric_val_best, epoch))
                state = {'epoch': epoch, 'it': it, 'loss_val_best': metric_val_best}
                state['state_dict'] = model.state_dict()
                torch.save(state, os.path.join(cfg['train']['out_dir'], 'model_best.pt'))

        # Save checkpoint
        if (epoch > 0) & (epoch % cfg['train']['checkpoint_every'] == 0):
            state = {'epoch': epoch,
                     'it': it,
                     'loss_val_best': metric_val_best}
            pcl = None
            state['state_dict'] = model.state_dict()
                
            torch.save(state, os.path.join(cfg['train']['out_dir'], 'model.pt'))

            if (it % cfg['train']['backup_every'] == 0):
                torch.save(state, os.path.join(cfg['train']['dir_model'], '%d' % epoch + '.pt'))
                logger.info("Backup model at epoch %d" % epoch)
            logger.info("Save new model at epoch %d" % epoch)

        done = time.time()


if __name__ == '__main__':
    main()
