import sys
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


def run_training(cfg):
    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = networks.PopulationNet(cfg.MODEL)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

    # reset the generators
    dataset = datasets.CellPopulationDataset(cfg=cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle':cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set, pop_set = [], []

        for i, batch in enumerate(dataloader):

            net.train()
            optimizer.zero_grad()

            x = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            y_pred = net(x)

            loss = criterion(y_pred, y_gts.float())
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            pop_set.append(y_gts.flatten())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation_cell(net, cfg, 'train', epoch_float, global_step, max_samples=1_000)
                evaluation.model_evaluation_cell(net, cfg, 'test', epoch_float, global_step, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                pop_set = torch.cat(pop_set)
                mean_pop = torch.mean(pop_set)
                null_percentage = torch.sum(pop_set == 0) / torch.numel(pop_set) * 100
                wandb.log({
                    'loss': np.mean(loss_set),
                    'labeled_percentage': 100,
                    'mean_population': mean_pop,
                    'null_percentage': null_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, pop_set = [], []

            if cfg.DEBUG:
                # testing evaluation
                evaluation.model_evaluation_census(net, cfg, 'nairobi')
                evaluation.model_evaluation_cell(net, cfg, 'train', epoch_float, global_step, max_samples=1_000)
                evaluation.model_evaluation_cell(net, cfg, 'test', epoch_float, global_step, max_samples=1_000)
                break
            # end of batch

        if not cfg.DEBUG:
            assert(epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

            # final evaluation
            evaluation.model_evaluation_cell(net, cfg, 'train', epoch_float, global_step)
            evaluation.model_evaluation_cell(net, cfg, 'test', epoch_float, global_step)
            for city in cfg.DATASET.CENSUS_EVALUATION_CITIES:
                print(f'Running census-level evaluation for {city}...')
                evaluation.model_evaluation_census(net, cfg, city)


if __name__ == '__main__':

    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        entity='population_mapping',
        project=args.project,
        tags=['run', 'population', 'mapping', 'regression', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
