import argparse
import mcubes
import multiprocessing
import numpy as np
import nrrd
import os
import open3d as o3d
from scipy import ndimage
import time
from tqdm import tqdm
import csv


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


def crop(complete, defective, implant):
    # Along x-axis
    idx = ~(complete == 0).all((1, 2))
    complete = complete[idx, :, :]
    defective = defective[idx, :, :]
    implant = implant[idx, :, :]

    # Along y-axis
    idx = ~(complete == 0).all((0, 2))
    complete = complete[:, idx, :]
    defective = defective[:, idx, :]
    implant = implant[:, idx, :]

    # Along z-axis
    idx = ~(complete == 0).all((0, 1))
    complete = complete[:, :, idx]
    defective = defective[:, :, idx]
    implant = implant[:, :, idx]

    return complete, defective, implant


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


def process_one_with_aug(filenames, p_rot=0.8, p_flip=0.5, p_noise=0.25, order=1, num_augments=10):

    for i in range(num_augments):
        # Load data
        complete = readCT(filenames['complete'])
        defective = readCT(filenames['defective'])
        implant = readCT(filenames['implant'])

        complete, defective, implant = crop(complete, defective, implant)
        complete = padding(complete)
        defective = padding(defective)
        implant = padding(implant)

        # Rotation
        rot = np.random.uniform(0, 1, 1)
        if rot < p_rot:
            rot_x, rot_y, rot_z = np.random.uniform(-10, 10, 3)

            complete = ndimage.rotate(complete, angle=rot_x, axes=(1, 2), order=order)
            complete = ndimage.rotate(complete, angle=rot_y, axes=(0, 2), order=order)
            complete = ndimage.rotate(complete, angle=rot_z, axes=(0, 1), order=order)

            defective = ndimage.rotate(defective, angle=rot_x, axes=(1, 2), order=order)
            defective = ndimage.rotate(defective, angle=rot_y, axes=(0, 2), order=order)
            defective = ndimage.rotate(defective, angle=rot_z, axes=(0, 1), order=order)

            implant = ndimage.rotate(implant, angle=rot_x, axes=(1, 2), order=order)
            implant = ndimage.rotate(implant, angle=rot_y, axes=(0, 2), order=order)
            implant = ndimage.rotate(implant, angle=rot_z, axes=(0, 1), order=order)

            complete, defective, implant = crop(complete, defective, implant)
            complete = padding(complete)
            defective = padding(defective)
            implant = padding(implant)

        # Flipping
        flip = np.random.uniform(0, 1, 1)
        if flip < p_flip:
            complete = np.flip(complete, axis=0)
            defective = np.flip(defective, axis=0)
            implant = np.flip(implant, axis=0)

        # Sample point cloud
        defective_vertices, defective_traingles = mcubes.marching_cubes(defective, 0)
        defective_mesh_name = os.path.join(filenames['defective'].split('/defective')[0], 'voxelization',
                                           filenames['complete'].split('.nrr')[0][-3:] + '_defective_' + str(i)
                                           + '.obj')
        mcubes.export_obj(defective_vertices, defective_traingles, defective_mesh_name)

        implant_vertices, implant_triangles = mcubes.marching_cubes(implant, 0)
        implant_mesh_name = os.path.join(filenames['defective'].split('/defective')[0], 'voxelization',
                                         filenames['complete'].split('.nrr')[0][-3:] + '_implant_' + str(i) + '.obj')
        mcubes.export_obj(implant_vertices, implant_triangles, implant_mesh_name)

        defective_surf = o3d.io.read_triangle_mesh(defective_mesh_name)
        defective_pc = defective_surf.sample_points_uniformly(number_of_points=27648)
        defective_pc_np = np.asarray(defective_pc.points)

        if os.path.exists(defective_mesh_name):
            os.remove(defective_mesh_name)

        implant_surf = o3d.io.read_triangle_mesh(implant_mesh_name)
        implant_pc = implant_surf.sample_points_uniformly(number_of_points=3072)
        implant_pc_np = np.asarray(implant_pc.points)

        if os.path.exists(implant_mesh_name):
            os.remove(implant_mesh_name)

        # Add noise
        add_noise = np.random.uniform(0, 1, 1)
        if add_noise < p_noise:
            noise = np.random.randn(*implant_pc_np.shape)
            noise = noise.astype(np.float32)
            implant_pc_np = implant_pc_np + noise

        pc = np.concatenate((defective_pc_np, implant_pc_np), axis=0)
        pc /= 512.0

        vox = np.ones((512, 512, 512), dtype=bool) * 0.5
        vox[complete > 0] = -0.5
        vox = vox.astype(np.float32)

        pc_name = os.path.join(filenames['defective'].split('/defective')[0], 'voxelization',
                               filenames['complete'].split('.nrr')[0][-3:] + str(i) + '_pc.npz')
        vox_name = os.path.join(filenames['defective'].split('/defective')[0], 'voxelization',
                                filenames['complete'].split('.nrr')[0][-3:] + str(i) + '_vox.npz')

        np.savez_compressed(pc_name, points=pc)
        np.savez_compressed(vox_name, psr=vox)


def main():
    print('---------------------------------------')
    print('Augment SkullFix dataset')
    print('---------------------------------------')

    dataset_folder = opt.path
    database = []

    if not os.path.isdir('datasets/SkullFix/voxelization'):
        os.makedirs('datasets/SkullFix/voxelization')

    with open(dataset_folder, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            datapoint = dict()
            datapoint['defective'] = row[0].split('complete')[0] + 'defective_skull/' + \
                                     row[0].split('skull')[1].split('.')[0] + '.nrrd'
            datapoint['implant'] = row[0].split('complete')[0] + 'implant/' + \
                                   row[0].split('skull')[1].split('.')[0] + '.nrrd'
            datapoint['complete'] = row[0]
            database.append(datapoint)

    if multiprocess:
        # multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(process_one_with_aug, database), total=len(database)):
                pass
            # pool.map_async(process_one, obj_list).get()
        except KeyboardInterrupt:
            # Allow ^C to interrupt from any thread.
            exit()
        pool.close()
    else:
        for obj in tqdm(database):
            process_one_with_aug(obj)

    print('Done Processing')


if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    print('Total processing time: ', t_end - t_start)
