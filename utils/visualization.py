from pathlib import Path
from utils import geofiles


def plot_vhr(ax, dataset_path: str, city: str, i: int, j: int, vis: str = 'true_color', scale_factor: float = 500):
    file = Path(dataset_path) / 'features' / city / 'vhr' / f'vhr_{city}_{i:03d}-{j:03d}.tif'
    img, _, _ = geofiles.read_tif(file)
    band_indices = [0, 1, 2] if vis == 'true_color' else [3, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_s2(ax, dataset_path: str, city: str, i: int, j: int, vis: str = 'true_color', scale_factor: float = 5_000):
    file = Path(dataset_path) / 'features' / city / 's2' / f's2_{city}_{i:03d}-{j:03d}.tif'
    img, _, _ = geofiles.read_tif(file)
    band_indices = [0, 1, 2] if vis == 'true_color' else [3, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_bf(ax, dataset_path: str, city: str, i: int, j: int):
    file = Path(dataset_path) / 'features' / city / 'bf' / f'bf_{city}_{i:03d}-{j:03d}.tif'
    img, _, _ = geofiles.read_tif(file)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
