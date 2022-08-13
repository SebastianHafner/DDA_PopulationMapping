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


def run_dual_training(dual_cfg: experiment_manager.CfgNode):
    run_config = {
        'CONFIG_NAME': dual_cfg.NAME,
        'device': device,
        'epochs': dual_cfg.TRAINER.EPOCHS,
        'learning rate': dual_cfg.TRAINER.LR,
        'batch size': dual_cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    dual_net = networks.DualStreamPopulationNet(dual_cfg.MODEL)
    dual_net.to(device)
    optimizer = optim.AdamW(dual_net.parameters(), lr=dual_cfg.TRAINER.LR, weight_decay=0.01)

    criterion_stream1 = loss_functions.get_criterion(dual_cfg.MODEL.STREAM1.LOSS_TYPE)
    criterion_stream2 = loss_functions.get_criterion(dual_cfg.MODEL.STREAM2.LOSS_TYPE)
    criterion_fusion = loss_functions.get_criterion(dual_cfg.MODEL.LOSS_TYPE)
    criterion_consistency = loss_functions.get_criterion(dual_cfg.CONSISTENCY_TRAINER.LOSS_TYPE)

    # reset the generators
    dataset = datasets.CellDualInputPopulationDataset(dual_cfg=dual_cfg, run_type='train')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': dual_cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if dual_cfg.DEBUG else dual_cfg.DATALOADER.NUM_WORKER,
        'shuffle': dual_cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = dual_cfg.TRAINER.EPOCHS
    save_checkpoints = dual_cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set_stream1, loss_set_stream2, loss_set_fusion, pop_set = [], [], [], []
        supervised_loss_set, consistency_loss_set, loss_set = [], [], []
        n_labeled, n_notlabeled = 0, 0

        for i, (batch) in enumerate(dataloader):

            dual_net.train()
            dual_net.zero_grad()

            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            gt = batch['y'].to(device).float()
            is_labeled = batch['is_labeled'].to(device)
            pred_fusion, pred_stream1, pred_stream2 = dual_net(x1, x2)

            supervised_loss, consistency_loss = None, None

            # supervised loss
            if is_labeled.any():
                loss_stream1 = criterion_stream1(pred_stream1[is_labeled], gt[is_labeled])
                loss_stream2 = criterion_stream2(pred_stream2[is_labeled], gt[is_labeled])
                loss_fusion = criterion_fusion(pred_fusion[is_labeled], gt[is_labeled])

                loss_set_stream1.append(loss_stream1.item())
                loss_set_stream2.append(loss_stream2.item())

                n_labeled += torch.sum(is_labeled).item()

                if not dual_cfg.MODEL.DISABLE_FUSION_LOSS:
                    supervised_loss = loss_stream1 + loss_stream2 + loss_fusion
                    loss_set_fusion.append(loss_fusion.item())
                else:
                    supervised_loss = loss_stream1 + loss_stream2
                    loss_set_fusion.append(0)

                supervised_loss_set.append(supervised_loss.item())

            # consistency loss for semi-supervised training
            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                n_notlabeled += torch.sum(not_labeled).item()

                consistency_loss = criterion_consistency(pred_stream1[not_labeled,], pred_stream2[not_labeled,])
                consistency_loss = dual_cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * consistency_loss
                consistency_loss_set.append(consistency_loss.item())

            if supervised_loss is None and consistency_loss is not None:
                loss = consistency_loss
            elif supervised_loss is not None and consistency_loss is not None:
                loss = supervised_loss + consistency_loss
            else:
                loss = supervised_loss

            loss.backward()
            optimizer.step()

            # loss_set_fusion.append(loss_fusion.item())
            loss_set.append(loss.item())
            pop_set.append(gt.flatten())

            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % dual_cfg.LOG_FREQ == 0 and not dual_cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'train', epoch_float, global_step,
                                                            max_samples=1_000)
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'test', epoch_float, global_step,
                                                            max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                labeled_percentage = n_labeled / (n_labeled + n_notlabeled) * 100
                pop_set = torch.cat(pop_set)
                mean_pop = torch.mean(pop_set)
                null_percentage = torch.sum(pop_set == 0) / torch.numel(pop_set) * 100
                wandb.log({
                    'loss': np.mean(loss_set),
                    'loss_stream1': np.mean(loss_set_stream1),
                    'loss_stream2': np.mean(loss_set_stream2),
                    'loss_fusion': np.mean(loss_set_fusion),
                    'supervised_loss': np.mean(supervised_loss_set),
                    'consistency_loss': np.mean(consistency_loss_set) if consistency_loss_set else 0,
                    'loss_set': np.mean(loss_set),
                    'labeled_percentage': labeled_percentage,
                    'mean_population': mean_pop,
                    'null_percentage': null_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set, pop_set = [], []

            if dual_cfg.DEBUG:
                # testing evaluation
                evaluation.model_evaluation_census_dualstream(dual_net, dual_cfg, 'dakar')
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'train', epoch_float, global_step,
                                                            max_samples=1_000)
                evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'test', epoch_float, global_step,
                                                            max_samples=1_000)
                break
            # end of batch

        if not dual_cfg.DEBUG:
            assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')

        if epoch in save_checkpoints and not dual_cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(dual_net, optimizer, epoch, global_step, dual_cfg)

            evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'train', epoch_float, global_step)
            evaluation.model_evaluation_cell_dualstream(dual_net, dual_cfg, 'test', epoch_float, global_step)
            for city in dual_cfg.DATASET.CENSUS_EVALUATION_CITIES:
                print(f'Running census-level evaluation for {city}...')
                evaluation.model_evaluation_census_dualstream(dual_net, dual_cfg, city)


if __name__ == '__main__':

    args = parsers.training_argument_parser().parse_known_args()[0]
    dual_cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(dual_cfg.SEED)
    np.random.seed(dual_cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: p', device)

    wandb.init(
        name=dual_cfg.NAME,
        config=dual_cfg,
        entity='population_mapping',
        project=args.project,
        tags=['run', 'population', 'mapping', 'regression', ],
        mode='online' if not dual_cfg.DEBUG else 'disabled',
    )

    try:
        run_dual_training(dual_cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
