import os
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
import nrrd
import time
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--num_points', type=int, default=30720, help="number of points the point cloud should contain")
parser.add_argument('--num_nn', type=int, default=3072, help="number of points that represent the implant")
parser.add_argument('--path', type=str, default='datasets/SkullBreak/skullbreak.csv')
opt = parser.parse_args()

multiprocess = opt.multiprocessing
njobs = opt.threads
save_pointcloud = True
save_psr_field = True
num_points = opt.num_points
num_nn = opt.num_nn
padding = 1.2
mesh_factor = 1.1


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
    gt_vox, _ = nrrd.read(obj['gt_vox'])  # Gt vox

    # Downsample point clouds
    num_pc = pc_np.shape[0]
    idx_pc = np.random.randint(0, num_pc, num_points - num_nn)
    pc_np = pc_np[idx_pc, :]

    num_pc_d = pc_d_np.shape[0]
    idx_pc_d = np.random.randint(0, num_pc_d, num_nn)
    pc_d_np = pc_d_np[idx_pc_d, :]

    points = np.concatenate((pc_np, pc_d_np), axis=0)  # Pc complete (defective skull + implant)

    vox = np.ones((512, 512, 512), dtype=bool) * 0.5
    vox[gt_vox > 0] = -0.5
    vox = vox.astype(np.float32)

    name_vox = os.path.join(obj['defective_skull'].split('/defective')[0], 'voxelization',
                            obj['gt_vox'].split('.nrr')[0][-3:] + '_' + obj['defect'] + '_vox.npz')
    name_points = os.path.join(obj['defective_skull'].split('/defective')[0], 'voxelization',
                               obj['gt_vox'].split('.nrr')[0][-3:] + '_' + obj['defect'] + '_pc.npz')

    # normalize pc
    max = 512.0
    points /= max

    if save_pointcloud:
        np.savez_compressed(name_points, points=points)
    if save_psr_field:
        np.savez_compressed(name_vox, psr=vox)


def main():

    print('---------------------------------------')
    print('Processing SkullBreak dataset')
    print('---------------------------------------')

    dataset_folder = opt.path
    defects = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']
    database = []

    if not os.path.isdir('datasets/SkullBreak/voxelization'):
        os.makedirs('datasets/SkullBreak/voxelization')

    with open(dataset_folder, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            for defect_id in range(5):
                datapoint = dict()
                datapoint['defective_skull'] = row[0].split('complete')[0] + 'defective_skull/' + \
                                               defects[defect_id] + row[0].split('skull')[1].split('.')[0] + '_surf.npy'
                datapoint['implant'] = row[0].split('complete')[0] + 'implant/' + \
                                       defects[defect_id] + row[0].split('skull')[1].split('.')[0] + '_surf.npy'
                datapoint['gt_vox'] = row[0]
                datapoint['defect'] = defects[defect_id]
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
