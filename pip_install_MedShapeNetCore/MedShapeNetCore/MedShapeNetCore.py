__version__ = "0.1.11"

import numpy as np
import trimesh
import SimpleITK as sitk
import scipy
import scipy.ndimage
import open3d as o3d
import matplotlib.pyplot as plt
import tempfile
import os
from collections import OrderedDict
from pymeshfix._meshfix import PyTMesh
import io
from glob import glob
import requests
import tarfile
from io import BytesIO



class MyDict(OrderedDict):
    def __missing__(self, key):
        val = self[key] = MyDict()


data_original={
    'facialVR':'https://files.icg.tugraz.at/f/33c88e8c4c5c414c8ca7/?dl=1',
    'VAT':'https://files.icg.tugraz.at/f/30c753ebdd9e4f6a8ee6/?dl=1',
    'ThoracicAorta_Saitta':'https://files.icg.tugraz.at/f/7fff5da8efb74fe8b14f/?dl=1',
    'CoronaryArteries':'https://files.icg.tugraz.at/f/c9b7552cc88c4e549ec6/?dl=1'
}


class BatchLoader:

    def __init__(
        self,
        dataset,
        file_format,
        batch_size=2,
        shuffle=True
        ):

        """
        Wrap the shape data into 
        a dataloader. Example:
 
        >>facial_point=BatchLoader('facialVR','point',batch_size=2,shuffle=True)
        >>for batch in facial_point:
              print(batch.shape)

        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.format=file_format
        self.shuffle = shuffle

        self.download_data()

        if self.format=='mask':
            directory='./'+str(dataset)+'/masks'
            print(directory)
            self.data_list=glob('{}/*.nii.gz'.format(directory))              

        elif self.format=='point':
            directory='./'+str(dataset)+'/point_cloud'
            self.data_list=glob('{}/*.ply'.format(directory))                 

        elif self.format=='mesh':
            directory='./'+str(dataset)+'/mesh'
            self.data_list=glob('{}/*.stl'.format(directory))   
        
        else:
            raise Exception(
                f"Sorry, no file type {self.format}", 
                "please choose from 'masks,point_cloud,mesh'."
                )

        if len(self.data_list)==0:
            raise RuntimeError(
                f"Data type {self.format} does not exist in "
                f"dataset {dataset}, please choose other "
                 "data types."
            )

            
    def __len__(self):
        return int(len(self.data_list) / self.batch_size)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        i = self.counter * self.batch_size
        self.counter += 1
        if len(self.data_list) < i+self.batch_size: 
            raise StopIteration

        if self.shuffle:
            import random
            random.shuffle(self.data_list)

        files = self.data_list[i:i+self.batch_size]
        result=[]

        for file in files:
            if self.format=='mask':
                mask_img = sitk.ReadImage(file)
                data = sitk.GetArrayFromImage(mask_img)

            if self.format=='point':
                data=trimesh.load(file).vertices

            if self.format=='mesh':
                mesh=trimesh.load(file)
                vertices=mesh.vertices
                faces=mesh.faces
                data={
                  'vertices':vertices,
                  'faces':faces
                }

            result.append(data)

        return np.array(result)


    def download_data(self):

        if not os.path.exists(f'./{self.dataset}'):

            try:
                print(f'downloading dataset {self.dataset}...')
                r = requests.get(data_original[self.dataset], stream=True)
                with open('temp.tar.gz', 'wb') as f:
                    f.write(r.content)
                print(f'downloading complete, unzipping to folder ./{self.dataset}')
                tar = tarfile.open("temp.tar.gz")
                tar.extractall()
                tar.close()

            except:
                    raise RuntimeError(
                    f"downloading dataset {self.dataset} failed, please download"
                    f"and unzip it manualy via the link below: \n"
                    f"{data_original[self.dataset]}"
                    )
        else:
            if len(os.listdir('./'+self.dataset))!=0: 
                print(f'dataset already exists in folder ./{self.dataset}')


class MSNLoader(object):
	
	def __init__(self):
		self.data_path='./medshapenetcore_npz/'

	def load(self,filename):
		path=self.data_path+'medshapenetcore'+'_'+str(filename)+'.npz'

		if not os.path.isfile(path):
			raise Exception(f"file {path} not available, please download it first")

		data=np.load(path,allow_pickle=True,mmap_mode='c')['data'].item()

		print('current dataset:',path)
		print('available keys in the dataset:', list(data.keys()))
		return data


	def fast_load(self,dataset,formats,num_sample,shuffle=False):

	    """
	    To load only a specified number of shapes of a 
	    given format in order to load faster and use less
	    RAM. This function is useful for a quick preview of
	    the dataset, without downloading it. Example:

	    >>data=fast_load('facialVR','point',2,True)
	    >>data=fast_load('facialVR','mesh',2,True)
	    >>data=fast_load('facialVR','mesh',2, True)
	    >>print(data[0]['vertices'].shape)
	    >>print(data[0]['faces'].shape)
	    """       

	    import random

	    data_list=[]
	    response = requests.get(data_original[dataset], stream=True)
	    if response.status_code == 200:
	        with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar:
	            counter=1
	            temp=tar.getmembers()
	            if shuffle:
	                import random
	                random.shuffle(temp)
	            for member in temp:
	                if member.isfile() and formats in member.name:
	                    if counter<=num_sample:
	                        temp_dir = tempfile.mkdtemp()
	                        tar.extract(member, path=temp_dir)
	                        file_path = os.path.join(temp_dir, member.name)
	                        data_list.append(file_path)
	                        counter+=1
	                    else:
	                        break
	            if len(data_list)!=0:
	                data_set=[]
	                for file in data_list:
	                    if formats=='mask':
	                        data = sitk.ReadImage(file)
	                        data = sitk.GetArrayFromImage(data)
	                    if formats=='point':
	                        data=trimesh.load(file).vertices
	                    if formats=='mesh':
	                        mesh=trimesh.load(file)
	                        vertices=mesh.vertices
	                        faces=mesh.faces
	                        data={
	                          'vertices':vertices,
	                          'faces':faces
	                        }
	                    data_set.append(data)

	                if len(data_list)==0:
	                    raise Exception(
	                        f"dataset {dataset} has now format {formats}, "
	                        "please try another format"
	                        )
	    else:
	        raise RuntimeError(
	            f"accessing dataset {dataset} failed,"
	             "please check your internet connection"
	            )

	    return np.array(data_set)



class MSNVisualizer(object):
	def __init__(self, figsize=(8,8)):
		self.figsize=figsize

	def plot_point(self,x,y,z):
		fig = plt.figure(figsize=self.figsize)
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(x, y, z)
		plt.show()

	def plot_mesh(self,verts,faces):
		mesh = trimesh.Trimesh(vertices=verts, faces=faces)
		mesh.show()

	def plot_mask_projection(self,mask):
		plt.subplot(131)
		plt.imshow(np.sum(mask,axis=0))
		plt.subplot(132)
		plt.imshow(np.sum(mask,axis=1))
		plt.subplot(133)
		plt.imshow(np.sum(mask,axis=2))
		plt.show()



class MSNSaver():
	def __init__(self,default_save_path='./medshapenetcore_saved/'):
		self.default_save_path=default_save_path
		if not os.path.exists(self.default_save_path):
			os.mkdir(self.default_save_path)

	def save_ply(self,points,filename):
		cloud=trimesh.PointCloud(points)
		cloud.export(self.default_save_path+filename)
		'''return a trimesh point cloud  object'''
		return cloud

	def save_stl(self,verts,faces,filename, watertight=True):
		mesh = trimesh.Trimesh(vertices=verts, faces=faces)
		mesh.fill_holes()
		'''the fill_holes() function does not always make the mesh water tight'''
		mesh.export(self.default_save_path+filename)
		'''return a trimesh mesh object'''
		if watertight==True:
			mfix_t = PyTMesh(False)
			vert=np.asarray(verts)
			facets=np.asarray(faces)
			mfix_t.load_array(vert, facets)
			mfix_t.save_file(self.default_save_path+'temp.stl')
			mfix = PyTMesh(False)
			mfix.load_file(self.default_save_path+'temp.stl')
			mfix.fill_small_boundaries(nbe=0, refine=True)
			v_new, f_new = mfix.return_arrays()
			mfix.save_file(self.default_save_path+filename)
			mesh = trimesh.Trimesh(vertices=v_new, faces=f_new)
		print('water tight mesh:',mesh.is_watertight)
		return mesh

	def save_nifti(self,nd_array,filename):
		siktImg=sitk.GetImageFromArray(nd_array)
		sitk.WriteImage(siktImg,self.default_save_path+filename)



class MSNTransformer(object):

	'''A variety of functions for 3D transformations'''

	def __init__(self,default_save_path='./medshapenetcore_saved/'):
		self.default_save_path=default_save_path
		if not os.path.exists(self.default_save_path):
			os.mkdir(self.default_save_path)

	def mask_downsampling(self,nd_array,target_dim):
		resized_mask=scipy.ndimage.zoom(
			nd_array,
			(
				target_dim[0]/nd_array.shape[0],
				target_dim[1]/nd_array.shape[1],
				target_dim[2]/nd_array.shape[2]
				),
			order=0,
			mode='nearest')
		return resized_mask


	def mesh_decimation(self, org_vert, org_face,target_num_faces=20000):
		print('original num vertices',len(org_vert))
		print('original num faces',len(org_face))		

		mesh = trimesh.Trimesh(vertices=org_vert, faces=org_face)
		mesh.fill_holes()
		mesh.export(self.default_save_path+'dec_temp.stl')
		mesh_in = o3d.io.read_triangle_mesh(self.default_save_path+'dec_temp.stl')
		mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=target_num_faces)
		mesh_smp = o3d.geometry.TriangleMesh.compute_triangle_normals(mesh_smp)
		verts_new=np.asarray(mesh_smp.vertices)
		faces_new=np.asarray(mesh_smp.triangles)
		o3d.io.write_triangle_mesh(self.default_save_path+'decimated_mesh.stl',mesh_smp)
		print('final num vertices',len(verts_new))
		print('final num faces',len(faces_new))		
		return mesh_smp,verts_new,faces_new

	def point_sampling_from_mesh(self,org_vert, org_face,target_num_points=20000):
		print('uniformly sampling points from a mesh surface... ')
		print('original num vertices',len(org_vert))
		print('original num faces',len(org_face))
		mesh = trimesh.Trimesh(vertices=org_vert, faces=org_face)		
		sampled_points, _= trimesh.sample.sample_surface_even(mesh,target_num_points)
		return sampled_points

	def mesh2voxel(self,org_vert,org_face, voxel_size=0.5, save_nifti=True):
		mesh = trimesh.Trimesh(vertices=org_vert, faces=org_face)
		mesh.export(self.default_save_path+'vox_temp.stl')
		m = trimesh.load(self.default_save_path+'vox_temp.stl')
		v = m.voxelized(pitch=voxel_size)
		voxel_grid=v.matrix+1-1
		print('size of the voxel grid',voxel_grid.shape)
		if save_nifti:
			mask_img = sitk.GetImageFromArray(voxel_grid)
			sitk.WriteImage(mask_img,self.default_save_path+'voxelized_temp.nii.gz')		
		return voxel_grid














