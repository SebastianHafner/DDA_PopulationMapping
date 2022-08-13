import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, experiment_manager, networks, geofiles
from scipy import stats
from pathlib import Path


class RegressionEvaluation(object):
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add_sample_numpy(self, pred: np.ndarray, label: np.ndarray):
        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def add_sample_torch(self, pred: torch.tensor, label: torch.tensor):
        pred = pred.float().detach().cpu().numpy()
        label = label.float().detach().cpu().numpy()
        self.add_sample_numpy(pred, label)

    def reset(self):
        self.predictions = []
        self.labels = []

    def root_mean_square_error(self) -> float:
        return np.sqrt(np.sum(np.square(np.array(self.predictions) - np.array(self.labels))) / len(self.labels))

    def r_square(self) -> float:
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.labels, self.predictions)
        return r_value


def model_evaluation_cell(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, run_type: str, epoch: float,
                          step: int, max_samples: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()
    dataset = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    max_samples = len(dataset) if max_samples is None else max_samples
    counter = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            img = batch['x'].to(device)
            label = batch['y'].to(device)
            pred = net(img)
            measurer.add_sample_torch(pred, label)
            counter += 1
            if counter == max_samples or cfg.DEBUG:
                break

    # assessment
    rmse = measurer.root_mean_square_error()
    print(f'RMSE {run_type} {rmse:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse,
        'step': step,
        'epoch': epoch,
    })


def model_evaluation_census(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, city: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    census = metadata['census']

    for unit_nr, unit_gt in tqdm(census.items()):
        unit_nr, unit_gt = int(unit_nr), int(unit_gt)
        ds = datasets.CensusPopulationDataset(cfg, city, unit_nr)
        if ds.split == 'test':
            # predict for all grids of a unit
            pred_total = 0
            for i, index in enumerate(range(len(ds))):
                item = ds.__getitem__(index)
                x = item['x'].to(device)
                pop_pred = net(x.to(device).unsqueeze(0))
                pred_total += pop_pred.cpu().item()
            measurer.add_sample_numpy(np.array(pred_total), np.array(unit_gt))

    rmse = measurer.root_mean_square_error()
    r_value = measurer.r_square()
    wandb.log({
        f'{city} rmse': rmse,
        f'{city} r2': r_value,
    })


def model_evaluation_cell_dualstream(dual_net: networks.DualStreamPopulationNet, dual_cfg: experiment_manager.CfgNode,
                                     run_type: str, epoch: float, step: int, max_samples: int = None):
    measurer_stream1 = RegressionEvaluation()
    measurer_stream2 = RegressionEvaluation()
    measurer_fusion = RegressionEvaluation()

    dataset = datasets.CellDualInputPopulationDataset(dual_cfg, run_type, no_augmentations=True,
                                                      include_unlabeled=False)

    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if dual_cfg.DEBUG else dual_cfg.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_net.to(device)
    max_samples = len(dataset) if max_samples is None else max_samples

    counter = 0
    with torch.no_grad():
        dual_net.eval()
        for step, batch in enumerate(tqdm(dataloader)):
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            label = batch['y'].to(device)

            pred_fusion, pred_stream1, pred_stream2 = dual_net(x1, x2)
            measurer_stream1.add_sample_torch(pred_stream1, label)
            measurer_stream2.add_sample_torch(pred_stream2, label)
            measurer_fusion.add_sample_torch(pred_fusion, label)

            counter += 1
            if counter == max_samples or dual_cfg.DEBUG:
                break

    # assessment
    rmse_stream1 = measurer_stream1.root_mean_square_error()
    rmse_stream2 = measurer_stream2.root_mean_square_error()
    rmse_fusion = measurer_fusion.root_mean_square_error()
    print(f'RMSE {run_type} {rmse_fusion:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse_stream1 if dual_cfg.MODEL.DISABLE_FUSION_LOSS else rmse_fusion,
        f'{run_type} rmse_stream1': rmse_stream1,
        f'{run_type} rmse_stream2': rmse_stream2,
        'step': step,
        'epoch': epoch,
    })


def model_evaluation_census_dualstream(dual_net: networks.DualStreamPopulationNet, dual_cfg: experiment_manager.CfgNode, city: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_net.to(device)
    dual_net.eval()

    measurer_stream1 = RegressionEvaluation()
    measurer_stream2 = RegressionEvaluation()
    measurer_fusion = RegressionEvaluation()

    metadata_file = Path(dual_cfg.PATHS.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    census = metadata['census']

    for unit_nr, unit_gt in tqdm(census.items()):
        unit_nr, unit_gt = int(unit_nr), int(unit_gt)
        ds = datasets.CensusDualInputPopulationDataset(dual_cfg, city, unit_nr)
        if ds.split == 'test':
            pred_total_fusion, pred_total_stream1, pred_total_stream2 = 0, 0, 0
            for i, index in enumerate(range(len(ds))):
                item = ds.__getitem__(index)
                x1 = item['x1'].to(device)
                x2 = item['x2'].to(device)
                pred_fusion, pred_stream1, pred_stream2 = dual_net(x1.to(device).unsqueeze(0), x2.to(device).unsqueeze(0))
                pred_total_fusion += pred_fusion.cpu().item()
                pred_total_stream1 += pred_stream1.cpu().item()
                pred_total_stream2 += pred_stream2.cpu().item()
            unit_gt = np.array([unit_gt])
            measurer_fusion.add_sample_numpy(np.array([pred_total_fusion]), unit_gt)
            measurer_stream1.add_sample_numpy(np.array([pred_total_stream1]), unit_gt)
            measurer_stream2.add_sample_numpy(np.array([pred_total_stream2]), unit_gt)

    rmse_fusion = measurer_fusion.root_mean_square_error()
    rmse_stream1 = measurer_stream1.root_mean_square_error()
    rmse_stream2 = measurer_stream2.root_mean_square_error()
    rsquare_fusion = measurer_fusion.r_square()
    rsquare_stream1 = measurer_stream1.r_square()
    rsquare_stream2 = measurer_stream2.r_square()

    wandb.log({
        f'{city} rmse': rmse_stream1 if dual_cfg.MODEL.DISABLE_FUSION_LOSS else rmse_fusion,
        f'{city} r2': rsquare_stream1 if dual_cfg.MODEL.DISABLE_FUSION_LOSS else rsquare_fusion,
        f'{city} rmse_stream1': rmse_stream1,
        f'{city} r2_stream1': rsquare_stream1,
        f'{city} rmse_stream2': rmse_stream2,
        f'{city} r2_stream2': rsquare_stream2,
    })
