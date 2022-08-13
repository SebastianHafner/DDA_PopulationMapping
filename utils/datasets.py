import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import affine
import math
import numpy as np
import cv2
from utils import augmentations, geofiles


class AbstractPopulationMappingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.patch_size = cfg.DATALOADER.PATCH_SIZE

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    # generic data loading function used for different features (e.g. vhr satellite data)
    def _get_patch_data(self, feature: str, city: str, i: int, j: int) -> np.ndarray:
        file = self.root_path / 'features' / city / feature / f'{feature}_{city}_{i:03d}-{j:03d}.tif'
        if feature == 'bf' and city == 'ouagadougou':
            img = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
        else:
            img, _, _ = geofiles.read_tif(file)

        if feature == 'vhr':
            band_indices = self.cfg.DATALOADER.VHR_BAND_INDICES
        elif feature == 's2':
            band_indices = self.cfg.DATALOADER.S2_BAND_INDICES
        else:
            band_indices = [0]
        img = img[:, :, band_indices]

        if feature == 'vhr':
            img = np.clip(img / self.cfg.DATALOADER.VHR_MAX_REFLECTANCE, 0, 1)
        if feature == 's2':
            img = np.clip(img / 10_000, 0, 1)

        # resampling images to desired patch size
        if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
                img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        return np.nan_to_num(img).astype(np.float32)

    # loading patch data for all features
    def _get_patch_net_input(self, features: list, city: str, i: int, j: int) -> np.ndarray:
        patch_data_features = []
        for feature in features:
            patch_data = self._get_patch_data(feature, city, i, j)
            patch_data_features.append(patch_data)
        patch_data_features = np.concatenate(patch_data_features, axis=-1)
        return patch_data_features.astype(np.single)

    @staticmethod
    def pop_log_conversion(pop: float) -> float:
        if pop == 0:
            return 0
        else:
            return math.log10(pop)


# dataset for urban extraction with building footprints
class CellPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, cfg, run_type: str, no_augmentations: bool = False, include_unlabeled: bool = True):
        super().__init__(cfg)

        self.run_type = run_type
        self.no_augmentations = no_augmentations
        self.include_unlabeled = include_unlabeled

        self.cities = list(cfg.DATASET.LABELED_CITIES)
        if cfg.DATALOADER.INCLUDE_UNLABELED:
            self.cities.extend(list(cfg.DATASET.UNLABELED_CITIES))

        self.samples = []
        for city in self.cities:
            city_metadata_file = self.root_path / f'metadata_{city}.json'
            city_metadata = geofiles.load_json(city_metadata_file)
            self.samples.extend(city_metadata['samples'])
        self.samples = [s for s in self.samples if s['split'] == run_type]

        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg.AUGMENTATION)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        if self.cfg.DATALOADER.LOG_POP:
            population = self.pop_log_conversion(float(sample['population']))
        else:
            population = float(sample['population'])

        patch_data = self._get_patch_net_input(self.cfg.DATALOADER.FEATURES, city, i, j)
        x = self.transform(patch_data)

        item = {
            'x': x,
            'y': torch.tensor([population]),
            'i': i,
            'j': j,
            'is_labeled': True if city in self.cfg.DATASET.LABELED_CITIES else False
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for urban extraction with building footprints
class CellInferencePopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, cfg, city: str):
        super().__init__(cfg)

        self.city = city
        self.samples = []
        metadata_file = self.root_path / f'metadata_{city}.json'
        metadata = geofiles.load_json(metadata_file)
        self.samples.extend(metadata['samples'])
        self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        patch_data = self._get_patch_net_input(self.cfg.DATALOADER.FEATURES, city, i, j)
        x = self.transform(patch_data)

        item = {
            'x': x,
            'i': i,
            'j': j,
        }

        return item

    def get_geo(self) -> tuple:
        sample = self.samples[0]
        i, j = sample['i'], sample['j']
        file = self.root_path / 'features' / self.city / 's2' / f's2_{self.city}_{i:03d}-{j:03d}.tif'
        _, transform, crs = geofiles.read_tif(file)
        x_min = transform[2] - j * 100
        y_min = transform[5] - i * -100
        transform = affine.Affine(100, 0, x_min, 0, -100, y_min)
        return transform, crs

    def get_arr(self) -> np.ndarray:
        m = sorted([s['i'] for s in self.samples])[-1] + 1
        n = sorted([s['j'] for s in self.samples])[-1] + 1
        arr = np.zeros((m, n, 1), dtype=np.uint16)
        return arr

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for urban extraction with building footprints
class CensusPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, cfg, city: str, unit_nr: int):
        super().__init__(cfg)

        self.unit_nr = unit_nr
        metadata_file = self.root_path / f'metadata_{city}.json'
        metadata = geofiles.load_json(metadata_file)
        self.unit_pop = metadata['census'][str(unit_nr)]
        self.split = metadata['split'][str(unit_nr)]
        all_samples = metadata['samples']
        self.samples = [s for s in all_samples if s['unit'] == unit_nr]

        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        self.length = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        population = float(sample['population'])

        patch_data = self._get_patch_net_input(self.cfg.DATALOADER.FEATURES, city, i, j)
        x = self.transform(patch_data)

        item = {
            'x': x,
            'y': torch.tensor([population]),
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for urban extraction with building footprints
class CellDualInputPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, dual_cfg, run_type: str, no_augmentations: bool = False, include_unlabeled: bool = True):
        super().__init__(dual_cfg)

        self.run_type = run_type
        self.no_augmentations = no_augmentations
        self.include_unlabeled = include_unlabeled

        self.cities = list(self.cfg.DATASET.LABELED_CITIES)
        if self.cfg.DATALOADER.INCLUDE_UNLABELED and include_unlabeled:
            self.cities.extend(list(self.cfg.DATASET.UNLABELED_CITIES))

        self.samples = []
        for city in self.cities:
            city_metadata_file = self.root_path / f'metadata_{city}.json'
            city_metadata = geofiles.load_json(city_metadata_file)
            samples = city_metadata['samples']
            if city in self.cfg.DATASET.LABELED_CITIES:
                samples = [s for s in samples if s['split'] == run_type]
            self.samples.extend(samples)

        if no_augmentations:
            self.transform_stream1 = transforms.Compose([augmentations.Numpy2Torch()])
            self.transform_stream2 = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform_stream1 = augmentations.compose_transformations(self.cfg.AUGMENTATION.STREAM1)
            self.transform_stream2 = augmentations.compose_transformations(self.cfg.AUGMENTATION.STREAM2)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        if self.cfg.DATALOADER.LOG_POP:
            population = self.pop_log_conversion(float(sample['population']))
        else:
            population = float(sample['population'])

        patch_data_stream1 = self._get_patch_data(self.cfg.DATALOADER.FEATURE_STREAM1, city, i, j)
        patch_data_stream2 = self._get_patch_data(self.cfg.DATALOADER.FEATURE_STREAM2, city, i, j)
        x1 = self.transform_stream1(patch_data_stream1)
        x2 = self.transform_stream1(patch_data_stream2)

        item = {
            'x1': x1,
            'x2': x2,
            'y': torch.tensor([population]),
            'i': i,
            'j': j,
            'is_labeled': True if city in self.cfg.DATASET.LABELED_CITIES else False
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for urban extraction with building footprints
class CensusDualInputPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, dual_cfg, city: str, unit_nr: int):
        super().__init__(dual_cfg)

        self.unit_nr = unit_nr
        metadata_file = self.root_path / f'metadata_{city}.json'
        metadata = geofiles.load_json(metadata_file)
        self.unit_pop = metadata['census'][str(unit_nr)]
        self.split = metadata['split'][str(unit_nr)]
        all_samples = metadata['samples']
        self.samples = [s for s in all_samples if s['unit'] == unit_nr]

        self.transform_stream1 = transforms.Compose([augmentations.Numpy2Torch()])
        self.transform_stream2 = transforms.Compose([augmentations.Numpy2Torch()])

        self.length = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        population = float(sample['population'])

        patch_data_stream1 = self._get_patch_data(self.cfg.DATALOADER.FEATURE_STREAM1, city, i, j)
        patch_data_stream2 = self._get_patch_data(self.cfg.DATALOADER.FEATURE_STREAM2, city, i, j)
        x1 = self.transform_stream1(patch_data_stream1)
        x2 = self.transform_stream1(patch_data_stream2)

        item = {
            'x1': x1,
            'x2': x2,
            'y': torch.tensor([population]),
            'i': i,
            'j': j,
            'is_labeled': True if city in self.cfg.DATASET.LABELED_CITIES else False
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for urban extraction with building footprints
class CellInferenceDualInputPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, cfg, city: str):
        super().__init__(cfg)

        self.city = city
        self.samples = []
        metadata_file = self.root_path / f'metadata_{city}.json'
        metadata = geofiles.load_json(metadata_file)
        self.samples.extend(metadata['samples'])
        self.transform_stream1 = transforms.Compose([augmentations.Numpy2Torch()])
        self.transform_stream2 = transforms.Compose([augmentations.Numpy2Torch()])
        self.length = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']

        patch_data_stream1 = self._get_patch_data(self.cfg.DATALOADER.FEATURE_STREAM1, city, i, j)
        patch_data_stream2 = self._get_patch_data(self.cfg.DATALOADER.FEATURE_STREAM2, city, i, j)
        x1 = self.transform_stream1(patch_data_stream1)
        x2 = self.transform_stream1(patch_data_stream2)

        item = {
            'x1': x1,
            'x2': x2,
            'i': i,
            'j': j,
        }

        return item

    def get_geo(self) -> tuple:
        sample = self.samples[0]
        i, j = sample['i'], sample['j']
        file = self.root_path / 'features' / self.city / 's2' / f's2_{self.city}_{i:03d}-{j:03d}.tif'
        _, transform, crs = geofiles.read_tif(file)
        x_min = transform[2] - j * 100
        y_min = transform[5] - i * -100
        transform = affine.Affine(100, 0, x_min, 0, -100, y_min)
        return transform, crs

    def get_arr(self) -> np.ndarray:
        m = sorted([s['i'] for s in self.samples])[-1] + 1
        n = sorted([s['j'] for s in self.samples])[-1] + 1
        arr = np.zeros((m, n, 1), dtype=np.uint16)
        return arr

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'