import multiprocessing
import time
import argparse
import numpy as np
import nrrd
import os
import open3d as o3d
import mcubes
import scipy
from tqdm import tqdm

# Path to the complete_skull folder of SkullFix dataset
directory = 'datasets/SkullFix/complete_skull'

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--keep_mesh', type=eval, default=False, help="save meshes True/False")
opt = parser.parse_args()

database = []
multiprocess = opt.multiprocessing
njobs = opt.threads
keep_meshes = opt.keep_mesh


def re_sample(image, current_spacing, new_spacing):
    """
    Funtion to resample an CT image to a new voxel size
    """
    resize_factor = current_spacing / new_spacing
    new_shape = image.shape * resize_factor
    new_shape = np.round(new_shape)
    actual_resize_factor = new_shape / image.shape
    new_spacing = current_spacing / actual_resize_factor

    image_resized = scipy.ndimage.zoom(image, actual_resize_factor, order=1)

    return image_resized, new_spacing


def readCT(filename):
    """
    Function to read a .nrrd file and sample to new voxel size
    """
    ct_data, ct_header = nrrd.read(filename)

    ct_spacing = np.asarray([ct_header['space directions'][0, 0],
                             ct_header['space directions'][1, 1],
                             ct_header['space directions'][2, 2]])

    ct_origin = np.asarray([ct_header['space origin'][0],
                            ct_header['space origin'][1],
                            ct_header['space origin'][2]])

    # if ct_spacing[2] > 0:
    #     num_slices = int(180 / ct_spacing[2])
    #     ct_data = ct_data[:, :, -num_slices:]

    # resample data to voxel size 0.45 x 0.45 x 0.45 mm
    if ct_spacing[2] > 0:
        ct_data_resampled, _ = re_sample(ct_data, ct_spacing, new_spacing=[0.45, 0.45, 0.45])

    else:
        ct_data_resampled = ct_data

    return ct_data_resampled


def crop(complete, defective, implant):
    """
    Function to crop all zero slices along x, y and z
    """
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

    assert complete.shape == defective.shape == implant.shape
    return complete, defective, implant


def padding(complete, defective, implant):
    """
    Function to zero pad the cropped volumes to an equal volume size of 512 x 512 x 512
    """
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
    defective = np.pad(defective, (dim_x, dim_y, dim_z), 'constant', constant_values=0)
    implant = np.pad(implant, (dim_x, dim_y, dim_z), 'constant', constant_values=0)

    assert complete.shape == defective.shape == implant.shape
    return complete, defective, implant


def process_one(obj):
    datapoint = obj
    # ------------------------------------------------------------
    # Perform marching cubes to extract the skull/ implant surface
    # ------------------------------------------------------------

    complete = readCT(datapoint['complete'])
    defective = readCT(datapoint['defect'])
    implant = readCT(datapoint['implant'])

    complete, defective, implant = crop(complete, defective, implant)
    complete, defective, implant = padding(complete, defective, implant)

    # For complete skull
    complete_surf_vert, complete_surf_triangles = mcubes.marching_cubes(complete, 0)
    complete_surf_filename = datapoint['complete'].split('.nrrd')[0] + '_surf.obj'
    mcubes.export_obj(complete_surf_vert, complete_surf_triangles, complete_surf_filename)

    # For defective skull
    defective_surf_vert, defective_surf_triangles = mcubes.marching_cubes(defective, 0)
    defective_surf_filename = datapoint['defect'].split('.nrrd')[0] + '_surf.obj'
    mcubes.export_obj(defective_surf_vert, defective_surf_triangles, defective_surf_filename)

    # For implant
    impl_surf_vert, impl_surf_triangles = mcubes.marching_cubes(implant, 0)
    impl_surf_filename = datapoint['implant'].split('.nrrd')[0] + '_surf.obj'
    mcubes.export_obj(impl_surf_vert, impl_surf_triangles, impl_surf_filename)

    # ---------------------------------------------
    # Create point clouds from these surface meshes
    # ---------------------------------------------
    # For complete skull
    complete_surf = o3d.io.read_triangle_mesh(complete_surf_filename)
    complete_pc = complete_surf.sample_points_poisson_disk(400000)
    complete_pc_np = np.asarray(complete_pc.points)
    complete_pc_filename = complete_surf_filename.split('.obj')[0] + '.npy'
    np.save(complete_pc_filename, complete_pc_np)

    # For defective skull
    defective_surf = o3d.io.read_triangle_mesh(defective_surf_filename)
    defective_pc = defective_surf.sample_points_poisson_disk(400000)
    defective_pc_np = np.asarray(defective_pc.points)
    defective_pc_filename = defective_surf_filename.split('.obj')[0] + '.npy'
    np.save(defective_pc_filename, defective_pc_np)

    # For implant
    impl_surf = o3d.io.read_triangle_mesh(impl_surf_filename)
    impl_pc = impl_surf.sample_points_poisson_disk(400000)
    impl_pc_np = np.asarray(impl_pc.points)
    impl_pc_filename = impl_surf_filename.split('.obj')[0] + '.npy'
    np.save(impl_pc_filename, impl_pc_np)

    # ---------------------------------------------
    # Delete .obj files
    # ---------------------------------------------
    if not keep_meshes:
        if os.path.exists(complete_surf_filename):
            os.remove(complete_surf_filename)
        if os.path.exists(defective_surf_filename):
            os.remove(defective_surf_filename)
        if os.path.exists(impl_surf_filename):
            os.remove(impl_surf_filename)


def main():
    # Gather available data
    for root, dirs, files in os.walk(directory, topdown=False):
        for filename in files:
            # Create 5 datapoints for each complete skull
            if filename.endswith('.nrrd'):
                datapoint = dict()
                datapoint['complete'] = os.path.join(root, filename)
                pardir = root.split('/complete')
                datapoint['defect'] = os.path.join(pardir[0], 'defective_skull', filename)
                datapoint['implant'] = os.path.join(pardir[0], 'implant', filename)
                database.append(datapoint)

    if multiprocess:
        pool = multiprocessing.Pool(njobs)
        try:
            for _ in tqdm(pool.imap_unordered(process_one, database), total=len(database)):
                pass
        except KeyboardInterrupt:
            exit()
        pool.close()

    else:
        for obj in tqdm(database):
            process_one(obj)


if __name__ == "__main__":
    print('Preprocess SkullFix dataset ...')
    t_start = time.time()
    main()
    t_end = time.time()
    print('Done. Total processing time: ', t_end - t_start)
