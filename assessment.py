import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from tqdm import tqdm
from scipy import stats
from utils import datasets, experiment_manager, networks, evaluation, geofiles, parsers
from pathlib import Path
FONTSIZE = 16


def qualitative_assessment_cell(cfg: experiment_manager.CfgNode, run_type: str = 'test', n_samples: int = 30,
                                scale_factor: float = 0.3):
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    plot_size = 2
    n_cols = 5
    n_rows = n_samples // n_cols
    if n_samples % n_cols != 0:
        n_rows += 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*plot_size, n_rows*plot_size))

    indices = np.random.randint(0, len(ds), n_samples)
    for index, item_index in enumerate(tqdm(indices)):
        item = ds.__getitem__(item_index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu().item()
        pop = item['y'].item()

        img = x.cpu().numpy().transpose((1, 2, 0))
        img = img[:, :, :3] if img.shape[-1] > 3 else img
        img = np.clip(img / scale_factor, 0, 1)

        i = index // n_cols
        j = index % n_cols
        ax = axs[i, j] if n_rows > 1 else axs[index]
        ax.imshow(img)
        ax.set_title(f'Pred: {pred_pop: .0f} - Pop: {pop:.0f}')
        ax.set_axis_off()
    out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'dakar_qualitative_assessment_{cfg.NAME}.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def correlation_cell(cfg: experiment_manager.CfgNode, city: str, run_type: str = 'test', scale: str = 'linear'):
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    preds, gts = [], []
    for i, index in enumerate(tqdm(range(len(ds)))):
        item = ds.__getitem__(index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu().item()
        preds.append(pred_pop)
        pop = item['y'].item()
        gts.append(pop)
        if i == 100:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Calculate the point density
    xy = np.vstack([gts, preds])
    z = stats.gaussian_kde(xy)(xy)
    markersize = 10
    ax.scatter(gts, preds, c=z, s=markersize, label='Cell')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gts, preds)
    x = np.array([0, 1_000])
    # ax.plot(x, slope * x + intercept, c='k')
    # place a text box in upper left in axes coords
    textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONTSIZE,
            verticalalignment='top')

    pop_max = 1_000
    line = ax.plot([0, pop_max], [0, pop_max], c='k', zorder=-1, label='1:1 line')
    if scale == 'linear':
        ticks = np.linspace(0, pop_max, 5)
        pop_min = 0
    else:
        ticks = [1, 10, 100, 1_000]
        ax.set_xscale('log')
        ax.set_yscale('log')
        pop_min = 1
    ax.set_xlim(pop_min, pop_max)
    ax.set_ylim(pop_min, pop_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{tick:.0f}' for tick in ticks], fontsize=FONTSIZE)
    ax.set_yticklabels([f'{tick:.0f}' for tick in ticks], fontsize=FONTSIZE)
    ax.set_xlabel('Ground Truth', fontsize=FONTSIZE)
    ax.set_ylabel('Prediction', fontsize=FONTSIZE)
    legend_elements = [
        lines.Line2D([0], [0], color='k', lw=1, label='1:1 Line'),
        lines.Line2D([0], [0], marker='.', color='w', markerfacecolor='k', label='Cell', markersize=markersize),
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, frameon=False, loc='upper center')
    out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'{city}_correlation_cell_{cfg.NAME}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()


def quantitative_assessment_cell(config_name: str, run_type: str = 'test'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    measurer = evaluation.RegressionEvaluation()

    for i, index in enumerate(tqdm(range(len(ds)))):
        item = ds.__getitem__(index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu()
        pop = item['y'].cpu()
        measurer.add_sample(pred_pop, pop)

        if i == 100:
            pass
    rmse = measurer.root_mean_square_error()
    print(f'RMSE: {rmse:.2f}')


def run_quantitative_assessment_census(cfg: experiment_manager.CfgNode, city: str, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    census = metadata['census']

    data = {}
    for unit_nr, unit_pop in census.items():
        unit_nr, unit_pop = int(unit_nr), int(unit_pop)
        ds = datasets.CensusPopulationDataset(cfg, city, unit_nr)
        unit_pred = 0
        unit_gt = 0
        for i, index in enumerate(range(len(ds))):
            item = ds.__getitem__(index)
            x = item['x']
            pop_pred = net(x.to(device).unsqueeze(0)).flatten().cpu()
            unit_pred += pop_pred.item()
            unit_gt += item['y'].cpu().item()

        print(f'ID: {unit_nr}: Unit pop: {unit_pop} - Pop GT: {unit_gt:.0f} - Pop Pred: {unit_pred:.0f}')
        data[str(unit_nr)] = {'ref': unit_pop, 'sum_gt': unit_gt, 'sum_pred': unit_pred, 'split': ds.split}

    out_file = Path(cfg.PATHS.OUTPUT) / 'predictions' / f'{cfg.NAME}_{run_type}_{city}.geojson'
    geofiles.write_json(out_file, data)


def run_quantitative_assessment_census_dualstream(dual_cfg: experiment_manager.CfgNode, city: str, run_type: str = 'test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, dual_cfg, device)
    net.eval()

    metadata_file = Path(dual_cfg.PATHS.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    census = metadata['census']

    data = {}
    for unit_nr, unit_pop in census.items():
        unit_nr, unit_pop = int(unit_nr), int(unit_pop)
        ds = datasets.CensusDualInputPopulationDataset(dual_cfg, city, unit_nr)
        unit_pred = 0
        unit_gt = 0
        for i, index in enumerate(range(len(ds))):
            item = ds.__getitem__(index)
            x1 = item['x1'].to(device).unsqueeze(0)
            x2 = item['x2'].to(device).unsqueeze(0)
            pred_fusion, pred_stream1, _ = net(x1, x2)
            pop_pred = pred_stream1 if dual_cfg.MODEL.DISABLE_FUSION_LOSS else pred_fusion
            unit_pred += pop_pred.flatten().cpu().item()
            unit_gt += item['y'].cpu().item()

        print(f'ID: {unit_nr}: Unit pop: {unit_pop} - Pop GT: {unit_gt:.0f} - Pop Pred: {unit_pred:.0f}')
        data[str(unit_nr)] = {'ref': unit_pop, 'sum_gt': unit_gt, 'sum_pred': unit_pred, 'split': ds.split}

    out_file = Path(dual_cfg.PATHS.OUTPUT) / 'predictions' / f'{dual_cfg.NAME}_{run_type}_{city}.geojson'
    geofiles.write_json(out_file, data)


def correlation_census(cfg: experiment_manager.CfgNode, city: str, run_type: str = 'test', scale: str = 'linear'):
    pred_file = Path(cfg.PATHS.OUTPUT) / 'predictions' / f'{cfg.NAME}_{run_type}_{city}.geojson'
    if not pred_file.exists():
        if cfg.MODEL.DUALSTREAM:
            run_quantitative_assessment_census_dualstream(cfg, city, run_type)
        else:
            run_quantitative_assessment_census(cfg, city, run_type)
    data = geofiles.load_json(pred_file)
    gts = [v['ref'] for v in data.values() if v['split'] == run_type]
    preds = [v['sum_pred'] for v in data.values() if v['split'] == run_type]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(gts, preds, c='k', s=10, label='Census area')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gts, preds)
    x = np.array([0, 1_000])
    ax.plot(x, slope * x + intercept, c='k')
    # place a text box in upper left in axes coords
    textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONTSIZE,
            verticalalignment='top')
    gt_max, pred_max = np.max(gts), np.max(preds)
    pop_max = gt_max if gt_max > pred_max else pred_max
    if pop_max < 1_000:
        pop_max = 1_000
    elif pop_max < 5_000:
        pop_max = 5_000
    elif pop_max < 10_000:
        pop_max = 10_000
    elif pop_max < 50_000:
        pop_max = 50_000
    elif pop_max < 100_000:
        pop_max = 100_000
    else:
        pop_max = 1_000_000

    ax.plot([0, pop_max], [0, pop_max], c='k', zorder=-1, label='1:1 line')

    if scale == 'linear':
        ticks = np.linspace(0, pop_max, 6)
        pop_min = 0
    else:
        ticks = [1, 10, 100, 1_000]
        ax.set_xscale('log')
        ax.set_yscale('log')
        pop_min = 1
    ax.set_xlim(pop_min, pop_max)
    ax.set_ylim(pop_min, pop_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{tick:.0f}' for tick in ticks], fontsize=FONTSIZE)
    ax.set_yticklabels([f'{tick:.0f}' for tick in ticks], fontsize=FONTSIZE)
    ax.set_xlabel('Ground Truth', fontsize=FONTSIZE)
    ax.set_ylabel('Prediction', fontsize=FONTSIZE)
    ax.legend(frameon=False, fontsize=FONTSIZE, loc='upper center')
    out_file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'{city}_correlation_census_{cfg.NAME}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()


def produce_population_grid(cfg: experiment_manager.CfgNode, city: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    if cfg.MODEL.DUALSTREAM:
        ds = datasets.CellInferenceDualInputPopulationDataset(cfg, city)
        arr = ds.get_arr()
        transform, crs = ds.get_geo()
        for item in ds:
            x1 = item['x1'].to(device).unsqueeze(0)
            x2 = item['x2'].to(device).unsqueeze(0)
            i, j = item['i'], item['j']
            pred_fusion, pred_stream1, _ = net(x1, x2)
            pop_pred = pred_stream1 if cfg.MODEL.DISABLE_FUSION_LOSS else pred_fusion
            arr[i, j, 0] = pop_pred.flatten().cpu().item()
    else:
        ds = datasets.CellInferencePopulationDataset(cfg, city)
        arr = ds.get_arr()
        transform, crs = ds.get_geo()
        for item in ds:
            x = item['x'].to(device)
            i, j = item['i'], item['j']
            pred_pop = net(x.unsqueeze(0)).flatten().cpu().item()
            arr[i, j, 0] = pred_pop
    out_file = Path(cfg.PATHS.OUTPUT) / 'population_grids' / f'pop_{city}_{cfg.NAME}.tif'
    geofiles.write_tif(out_file, arr, transform, crs)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    for city in args.sites:
        produce_population_grid(cfg, city)
        correlation_census(cfg, city)
