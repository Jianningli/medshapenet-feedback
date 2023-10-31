import os
import logging
import torch
import open3d as o3d
from torch.utils import data
from pdb import set_trace as st
from src.dpsr import DPSR
import csv
import trimesh
import numpy as np
import yaml

logger = logging.getLogger(__name__)

resolution = 512    # Poisson indicator grid resolution
padding = 1.2
dpsr = DPSR(res=(resolution, resolution, resolution), sig=0)


def noise_pc(points, stddev):
    noise = stddev * np.random.randn(*points.shape)
    noise = noise.astype(np.float32)

    return points + noise


def outliers_pc(points, ratio):
    n_points = points.shape[0]
    n_outlier_points = int(n_points * ratio)
    ind = np.random.randint(0, n_points, n_outlier_points)

    outliers = np.random.uniform(-0.55, 0.55, (n_outlier_points, 3))
    outliers = outliers.astype(np.float32)
    points[ind] = outliers

    return points

# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class SkullEval(data.Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.database = []

        for root, dirs, files in os.walk(self.dataset_folder):
            for dirname in dirs:
                if dirname.endswith('_surf'):
                    self.database.append(os.path.join(root, dirname))

    def __len__(self):
        return len(self.database)

    def __getitem__(self, item):
        data_dir = self.database[item]
        data = dict()

        samples = np.load(os.path.join(data_dir, 'sample.npy'))
        scale = np.load(os.path.join(data_dir, 'scale.npy'))[0]
        shift = np.load(os.path.join(data_dir, 'shift.npy'))[0]

        # Rescale to (0, 512) full size
        samples = samples * scale + shift

        # Scale to (0,1) space
        samples /= 512

        data['name'] = data_dir
        data['inputs'] = torch.from_numpy(samples).float()

        return(data)


class SkullDataset(data.Dataset):
    def __init__(self, path, split, noise_stddev=None, outlier_ratio=None):
        self.split = split
        self.path = path
        self.database = []
        self.noise_stddev = noise_stddev
        self.outlier_ratio = outlier_ratio

        with open(path, 'r', newline='') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                datapoint = dict()
                datapoint['pointcloud'] = '../' + row[0] + '_pc.npz'
                datapoint['gt_psr'] = '../' + row[0] + '_vox.npz'
                self.database.append(datapoint)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, item):
        data_dir = self.database[item]
        data = dict()

        pcd = np.load(data_dir['pointcloud'])
        pc = pcd['points']

        vox = np.load(data_dir['gt_psr'])
        psr_gt = vox['psr']

        if self.noise_stddev:
            pc = noise_pc(pc, self.noise_stddev)

        if self.outlier_ratio:
            pc = outliers_pc(pc, self.outlier_ratio)

        data['inputs'] = torch.from_numpy(pc).float()
        data['gt_psr'] = psr_gt

        return data


class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None, cfg=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f, Loader=yaml.Loader)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            if split is None:
                self.models += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]
        
        # precompute
        self.split = split
            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}

        info = c_idx
        
        if self.cfg['data']['multi_files'] is not None:
            idx = np.random.randint(self.cfg['data']['multi_files'])
            if self.split != 'train':
                idx = 0

        for field_name, field in self.fields.items():
            try:
                field_data = field.load(model_path, idx, info)
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of model %s'
                        % (field_name, model)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data
        
    
    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)

def collate_stack_together(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    keys = batch[0].keys()
    concat = {}
    if len(batch)>1:
        for key in keys:
            key_val = [item[key] for item in batch]
            concat[key] = np.concatenate(key_val, axis=0)
            if key == 'inputs':
                n_pts = [item[key].shape[0] for item in batch]

        concat['batch_ind'] = np.concatenate(
                [i * np.ones(n, dtype=int) for i, n in enumerate(n_pts)], axis=0)

        return data.dataloader.default_collate([concat])
    else:
        n_pts = batch[0]['inputs'].shape[0]
        batch[0]['batch_ind'] = np.zeros(n_pts, dtype=int)
        return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    def set_num_threads(nt):
        try: 
            import mkl; mkl.set_num_threads(nt)
        except: 
            pass
            torch.set_num_threads(1)
            os.environ['IPC_ENABLE']='1'
            for o in ['OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
