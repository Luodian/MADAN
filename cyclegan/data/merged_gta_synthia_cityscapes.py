import os.path
import random

import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_label_transform, get_transform
from data.image_folder import make_cs_labels, make_dataset

from data.cityscapes import remap_labels_to_train_ids

ignore_label = 255
id2label = {0: ignore_label,
            1: 10,
            2: 2,
            3: 0,
            4: 1,
            5: 4,
            6: 8,
            7: 5,
            8: 13,
            9: 7,
            10: 11,
            11: 18,
            12: 17,
            13: ignore_label,
            14: ignore_label,
            15: 6,
            16: 9,
            17: 12,
            18: 14,
            19: 15,
            20: 16,
            21: 3,
            22: ignore_label}

classes = ['road',
           'sidewalk',
           'building',
           'wall',
           'fence',
           'pole',
           'traffic light',
           'traffic sign',
           'vegetation',
           'terrain',
           'sky',
           'person',
           'rider',
           'car',
           'truck',
           'bus',
           'train',
           'motorcycle',
           'bicycle']


def syn_relabel(arr):
	out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
	for id, label in id2label.items():
		out[arr == id] = int(label)
	return out

# In this dataset, we are using one cyclegan to iteratively conduct GTAV->CityScapes and Synthia->CityScapes
class MergedGTASynthiaCityscapesDataset(BaseDataset):
	def initialize(self, opt):
		# SYNTHIA as dataset 1
		# GTAV as dataset 2
		self.opt = opt
		self.root = opt.dataroot
		self.dir_A_1 = os.path.join(opt.dataroot, 'synthia', 'RGB')
		self.dir_A_2 = os.path.join(opt.dataroot, 'gta5', 'images')
		self.dir_B = os.path.join(opt.dataroot, 'cityscapes', 'leftImg8bit')
		self.dir_A_label_1 = os.path.join(opt.dataroot, 'synthia', 'GT', 'parsed_LABELS')
		self.dir_A_label_2 = os.path.join(opt.dataroot, 'gta5', 'labels')
		self.dir_B_label = os.path.join(opt.dataroot, 'cityscapes', 'gtFine')
		
		self.A_paths_1 = make_dataset(self.dir_A_1)
		self.A_paths_2 = make_dataset(self.dir_A_2)
		self.B_paths = make_dataset(self.dir_B)
		
		self.A_paths_1 = sorted(self.A_paths_1)
		self.A_paths_2 = sorted(self.A_paths_2)
		
		self.B_paths = sorted(self.B_paths)
		
		self.A_size_1 = len(self.A_paths_1)
		self.A_size_2 = len(self.A_paths_2)
		
		self.B_size = len(self.B_paths)
		
		self.A_labels_1 = make_dataset(self.dir_A_label_1)
		self.A_labels_2 = make_dataset(self.dir_A_label_2)
		
		self.B_labels = make_cs_labels(self.dir_B_label)
		
		self.A_labels_1 = sorted(self.A_labels_1)
		self.A_labels_2 = sorted(self.A_labels_2)
		self.B_labels = sorted(self.B_labels)
		
		self.transform = get_transform(opt)
		self.label_transform = get_label_transform(opt)
	
	def __getitem__(self, index):
		if self.opt.serial_batches:
			index_B = index % self.B_size
		else:
			index_B = random.randint(0, self.B_size - 1)
		
		B_path = self.B_paths[index_B]
		B_label_path = self.B_labels[index_B]
		B_label = Image.open(B_label_path)
		B_label = np.asarray(B_label)
		B_label = remap_labels_to_train_ids(B_label)
		B_label = Image.fromarray(B_label, 'L')
		B_img = Image.open(B_path).convert('RGB')
		B = self.transform(B_img)
		B_label = self.label_transform(B_label)
		
		if self.opt.which_direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc
		
		if index % 2:
			A_path_1 = self.A_paths_1[index % self.A_size_1]
			A_label_path_1 = self.A_labels_1[index % self.A_size_1]
			A_label_1 = Image.open(A_label_path_1)
			# remaping label for synthia
			A_label_1 = np.asarray(A_label_1)
			A_label_1 = syn_relabel(A_label_1)
			A_label_1 = Image.fromarray(A_label_1, 'L')
			A_img_1 = Image.open(A_path_1).convert('RGB')
			A_1 = self.transform(A_img_1)
			A_label_1 = self.label_transform(A_label_1)
			return {'A': A_1, 'B': B, 'A_paths': A_path_1, 'B_paths': B_path, 'A_label': A_label_1, 'B_label': B_label}
		else:
			A_path_2 = self.A_paths_2[index % self.A_size_2]
			A_label_path_2 = self.A_labels_2[index % self.A_size_2]
			A_label_2 = Image.open(A_label_path_2)
			# remaping label for gta5
			A_label_2 = np.asarray(A_label_2)
			A_label_2 = remap_labels_to_train_ids(A_label_2)
			A_label_2 = Image.fromarray(A_label_2, 'L')
			A_img_2 = Image.open(A_path_2).convert('RGB')
			A_2 = self.transform(A_img_2)
			A_label_2 = self.label_transform(A_label_2)
			return {'A': A_2, 'B': B, 'A_paths': A_path_2, 'B_paths': B_path, 'A_label': A_label_2, 'B_label': B_label}
	
	def __len__(self):
		return self.A_size_1 + self.A_size_2
	
	def name(self):
		return 'MergedGTASynthiaCityscapesDataset'
