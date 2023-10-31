import os
import argparse
import multiprocessing
import numpy as np
import nrrd
import time
import csv
from tqdm import tqdm
from scipy import ndimage

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--num_points', type=int, default=30720, help="number of points the point cloud should contain")
parser.add_argument('--num_nn', type=int, default=3072, help="number of points that represent the implant")
parser.add_argument('--path', type=str, default='datasets/SkullFix/skullfix.csv')
opt = parser.parse_args()

multiprocess = opt.multiprocessing
njobs = opt.threads
save_pointcloud = True
save_psr_field = True
num_points = opt.num_points
num_nn = opt.num_nn
padding = 1.2
mesh_factor = 1.1


def re_sample(image, current_spacing, new_spacing):
    resize_factor = current_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    actual_resize_factor = new_shape / image.shape
    new_spacing = current_spacing / actual_resize_factor

    image_resized = ndimage.zoom(image, actual_resize_factor)

    return image_resized, new_spacing


def readCT(filename):
    ct_data, ct_header = nrrd.read(filename)

    ct_spacing = np.asarray([ct_header['space directions'][0, 0],
                             ct_header['space directions'][1, 1],
                             ct_header['space directions'][2, 2]])

    ct_origin = np.asarray([ct_header['space origin'][0],
                            ct_header['space origin'][1],
                            ct_header['space origin'][2]])

    if ct_spacing[2] > 0:
        num_slices = int(180 / ct_spacing[2])
        ct_data = ct_data[:, :, -num_slices:]

        # resample data
    if ct_spacing[2] > 0:
        ct_data_resampled, _ = re_sample(ct_data, ct_spacing, new_spacing=[0.45, 0.45, 0.45])

    else:
        ct_data_resampled = ct_data

    return ct_data_resampled


def crop(complete):
    # Along x-axis
    idx = ~(complete == 0).all((1, 2))
    complete = complete[idx, :, :]

    # Along y-axis
    idx = ~(complete == 0).all((0, 2))
    complete = complete[:, idx, :]

    # Along z-axis
    idx = ~(complete == 0).all((0, 1))
    complete = complete[:, :, idx]

    return complete


def padding(complete):
    x = (512 - complete.shape[0]) / 2
    if x % 1 == 0:
        dim_x = (int(x), int(x))
    elif x == 0:
        dim_x = (0, 0)
    else:
        dim_x = (int(x-0.5), int(x+0.5))

    y = (512 - complete.shape[1]) / 2
    if y % 1 == 0:
        dim_y = (int(y), int(y))
    elif y == 0:
        dim_y = (0, 0)
    else:
        dim_y = (int(y-0.5), int(y+0.5))

    z = 512 - complete.shape[2]
    if z > 0:
        dim_z = (0, int(z))
    else:
        dim_z = (0, 0)

    complete = np.pad(complete, (dim_x, dim_y, dim_z), 'constant', constant_values=0)

    return complete


def array2voxel(voxel_array):
    """
    convert a to a fixed size array to voxel_grid_index array
    (voxel_size*voxel_size*voxel_size)->(N*3)

    :input voxel_array: array with shape(voxel_size*voxel_size*voxel_size),the grid_index in
    :return grid_index_array: get from o3d.voxel_grid.get_voxels()
    """
    x, y, z = np.where(voxel_array == 1)
    index_voxel = np.vstack((x, y, z))
    grid_index_array = index_voxel.T
    return grid_index_array


def process_one(obj):
    pc_np = np.load(obj['defective_skull'])  # Defective skull pc
    pc_d_np = np.load(obj['implant'])  # Implant pc
    gt_vox = readCT(obj['gt_vox'])  # Gt vox

    # Downsample point clouds
    num_pc = pc_np.shape[0]
    idx_pc = np.random.randint(0, num_pc, num_points - num_nn)
    pc_np = pc_np[idx_pc, :]

    num_pc_d = pc_d_np.shape[0]
    idx_pc_d = np.random.randint(0, num_pc_d, num_nn)
    pc_d_np = pc_d_np[idx_pc_d, :]

    points = np.concatenate((pc_np, pc_d_np), axis=0)  # Pc complete (defective skull + implant)
    gt_vox = crop(gt_vox)
    gt_vox = padding(gt_vox)

    vox = np.ones((512, 512, 512), dtype=bool) * 0.5
    vox[gt_vox > 0] = -0.5
    vox = vox.astype(np.float32)

    name_vox = os.path.join(obj['defective_skull'].split('/defective')[0], 'voxelization',
                            obj['gt_vox'].split('.nrr')[0][-3:] + '_vox.npz')
    name_points = os.path.join(obj['defective_skull'].split('/defective')[0], 'voxelization',
                               obj['gt_vox'].split('.nrr')[0][-3:] + '_pc.npz')

    # normalize pc
    max = 512.0
    points /= max

    if save_pointcloud:
        np.savez_compressed(name_points, points=points)
    if save_psr_field:
        np.savez_compressed(name_vox, psr=vox)


def main():

    print('---------------------------------------')
    print('Processing SkullFix dataset')
    print('---------------------------------------')

    dataset_folder = opt.path
    database = []

    if not os.path.isdir('datasets/SkullFix/voxelization'):
        os.makedirs('datasets/SkullFix/voxelization')

    with open(dataset_folder, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            datapoint = dict()
            datapoint['defective_skull'] = row[0].split('complete')[0] + 'defective_skull' +\
                                           row[0].split('skull')[1].split('.')[0] + '_surf.npy'
            datapoint['implant'] = row[0].split('complete')[0] + 'implant' +\
                                   row[0].split('skull')[1].split('.')[0] + '_surf.npy'
            datapoint['gt_vox'] = row[0]
            database.append(datapoint)

    if multiprocess:
        # multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(process_one, database), total=len(database)):
                pass
            # pool.map_async(process_one, obj_list).get()
        except KeyboardInterrupt:
            # Allow ^C to interrupt from any thread.
            exit()
        pool.close()
    else:
        for obj in tqdm(database):
            process_one(obj)

    print('Done Processing')


if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    print('Total processing time: ', t_end - t_start)