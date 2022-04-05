import os
import time
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed, set_logger, set_device, CheckpointIO, Trainer
import logging
logger_py = logging.getLogger(__name__)

def main(config):
    # set random seed and running device
    set_seed(config.seed)
    # set device
    device = set_device(config.device)

    # Directories and Shorthands
    out_dir = config.out_dir
    vis_dir = os.path.join(out_dir, 'vis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(vis_dir)

    print_every = config.print_every
    checkpoint_every = config.checkpoint_every
    validate_every = config.validate_every
    visualize_every = config.visualize_every
    backup_every = config.backup_every
    exit_after = config.exit_after

    model_selection_metric = config.model_selection_metric
    if config.model_selection_mode == 'maximize':
        model_selection_sign = 1
    elif config.model_selection_mode == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')
    # Prepare data or dataloader
    train_loader = None
    val_loader = None
    fixed_data = next(iter(val_loader))
    test_loader = None

    # Prepare model
    model = None
    pass

    # Initialize training
    optimizer = None
    trainer = Trainer(model, optimizer, device=device, vis_dir=vis_dir, overwrite_visualization=False)
    pass
    checkpoint_io = CheckpointIO(config.out_dir, model=model, optimizer=optimizer)

    try:
        load_dict = checkpoint_io.load('model.pt')
        print("Loaded model checkpoint.")
    except FileExistsError:
        load_dict = dict()
        print("No model checkpoint found.")

    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    print('Current best validation metric (%s): %.8f'
          % (model_selection_metric, metric_val_best))

    # set logger and tensorboard summarywriter
    set_logger(config)
    logger = SummaryWriter(os.path.join(config.out_dir, 'logs'))

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    logger_py.info(model)
    logger_py.info('Total number of parameters: %d' % nparameters)
    t0 = time.time()
    t0b = time.time()
    while True:
        epoch_it += 1

        for batch in train_loader:

            it += 1
            loss = trainer.train_step(batch, it)
            for (k, v) in loss.items():
                logger.add_scalar(k, v, it)
            # Print output
            if print_every > 0 and (it % print_every) == 0:
                info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (
                    epoch_it, it, time.time() - t0b)
                for (k, v) in loss.items():
                    info_txt += ', %s: %.4f' % (k, v)
                logger_py.info(info_txt)
                t0b = time.time()

            # # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                logger_py.info('Visualizing')
                print("Visulization...")
                image_grid = trainer.visualize(fixed_data, it=it)
                if image_grid is not None:
                    logger.add_image('images', image_grid[0], it)

            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                logger_py.info('Saving checkpoint')
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                logger_py.info('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Run validation
            if validate_every > 0 and (it % validate_every) == 0 and (it > 0):
                print("Performing evaluation step.")
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                logger_py.info('Validation metric (%s): %.4f'
                               % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    logger_py.info('New best model (%s %.4f)' % (model_selection_metric, metric_val_best))
                    checkpoint_io.backup_model_best('model_best.pt')
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)

            # Exit if necessary
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                logger_py.info('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                exit(3)

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Model description.')
    # Overall configuration.
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', type=int, default=-1, help='device number, device=-1 use cpu')

    # Model configuration.
    parser.add_argument('--model_selection_metric', type=str, default=None, choices=['psnr', 'ssim'])
    parser.add_argument('--model_selection_mode', type=str, default='maximize', choices=['maximize', 'minimize'])

    # Data and Directories configuration
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--val_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='output')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)

    # Shorthands
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=500)
    parser.add_argument('--validate_every', type=int, default=10000)
    parser.add_argument('--visualize_every', type=int, default=1000)
    parser.add_argument('--backup_every', type=int, default=1000000)
    parser.add_argument('--exit-after', type=int, default=-1,
                        help='Checkpoint and exit after specified number of '
                             'seconds with exit code 2.')

    config = parser.parse_args()

    if config.debug:
        config.batch_size = 2
        config.length = 10
        config.print_every = 1
        config.checkpoint_every = 1
        config.validate_every = 1
        config.visualize_every = 1
        config.backup_every = 1
    main(config)