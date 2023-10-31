import multiprocessing
import time
import argparse
import numpy as np
import nrrd
import os
import open3d as o3d
import mcubes
from tqdm import tqdm

# Path to the complete_skull folder of SkullBreak dataset
directory = 'datasets/SkullBreak/complete_skull'
defects = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--threads', type=int, default=8, help="define number of threads")
parser.add_argument('--keep_mesh', type=eval, default=False, help="save meshes True/False")
opt = parser.parse_args()

database = []
multiprocess = opt.multiprocessing
njobs = opt.threads
keep_meshes = opt.keep_mesh


def process_one(obj):
    datapoint = obj
    # ------------------------------------------------------------
    # Perform marching cubes to extract the skull/ implant surface
    # ------------------------------------------------------------
    # For complete skull
    if 'complete' in datapoint:
        complete, _ = nrrd.read(datapoint['complete'])
        complete_surf_vert, complete_surf_triangles = mcubes.marching_cubes(complete, 0)
        complete_surf_filename = datapoint['complete'].split('.nrrd')[0] + '_surf.obj'
        mcubes.export_obj(complete_surf_vert, complete_surf_triangles, complete_surf_filename)

    # For defective skull
    defective, _ = nrrd.read(datapoint['defect'])
    defective_surf_vert, defective_surf_triangles = mcubes.marching_cubes(defective, 0)
    defective_surf_filename = datapoint['defect'].split('.nrrd')[0] + '_surf.obj'
    mcubes.export_obj(defective_surf_vert, defective_surf_triangles, defective_surf_filename)

    # For ground truth implant
    impl, _ = nrrd.read(datapoint['implant'])
    impl_surf_vert, impl_surf_triangles = mcubes.marching_cubes(impl, 0)
    impl_surf_filename = datapoint['implant'].split('.nrrd')[0] + '_surf.obj'
    mcubes.export_obj(impl_surf_vert, impl_surf_triangles, impl_surf_filename)

    # ---------------------------------------------
    # Create point clouds from these surface meshes
    # ---------------------------------------------
    # For complete skull
    if 'complete' in datapoint:
        complete_surf = o3d.io.read_triangle_mesh(complete_surf_filename)
        complete_pc = complete_surf.sample_points_poisson_disk(400000)
        complete_pc_np = np.asarray(complete_pc.points)
        complete_pc_filename = complete_surf_filename.split('.obj')[0] + '.npy'
        np.save(complete_pc_filename, complete_pc_np)

    # For defective_skull
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

    if not keep_meshes:
        if 'complete' in datapoint:
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
                for defect_id in range(5):
                    datapoint = dict()
                    # Complete skull is only added once
                    if defect_id == 0:
                        datapoint['complete'] = os.path.join(root, filename)
                    pardir = root.split('/complete')
                    datapoint['defect'] = os.path.join(pardir[0], 'defective_skull', defects[defect_id], filename)
                    datapoint['implant'] = os.path.join(pardir[0], 'implant', defects[defect_id], filename)
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
    print('Preprocess SkullBreak dataset ...')
    t_start = time.time()
    main()
    t_end = time.time()
    print('Done. Total processing time: ', t_end - t_start)
