__version__ = "0.1.10"

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


class MyDict(OrderedDict):
    def __missing__(self, key):
        val = self[key] = MyDict()


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


	'''
	def load_as_pytroch_dataloader(self,filename,batchsize=1):

		print('loading the mask and point data as PyTorch Dataloader')

		import torch
		from torch.utils.data import TensorDataset, DataLoader

		data=self.load(filename)
		volumes=np.expand_dims(data['mask'],axis=1)
		points=np.expand_dims(data['point'],axis=1)
		
		tensor_volumes = torch.Tensor(volumes) 
		torch_dataset_volumes = TensorDataset(tensor_volumes) 
		dataloader_volumes = DataLoader(torch_dataset_volumes,batch_size=batchsize) 

		tensor_points = torch.Tensor(points) 
		torch_dataset_points = TensorDataset(tensor_points) 
		dataloader_points = DataLoader(torch_dataset_points,batch_size=batchsize) 
		return dataloader_volumes,dataloader_points
	'''

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














