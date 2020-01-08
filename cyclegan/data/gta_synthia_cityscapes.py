import os.path
import random

import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_label_transform, get_transform
from data.cityscapes import remap_labels_to_train_ids
from data.image_folder import make_cs_labels, make_dataset

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

# This dataset is used to conduct double cyclegan for both GTAV->CityScapes and Synthia->CityScapes
class GTASynthiaCityscapesDataset(BaseDataset):
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
		
		self.A_labels_1 = sorted(self.A_labels_1)
		self.A_labels_2 = sorted(self.A_labels_2)
		
		self.transform = get_transform(opt)
		self.label_transform = get_label_transform(opt)
	
	def __getitem__(self, index):
		A_path_1 = self.A_paths_1[index % self.A_size_1]
		A_path_2 = self.A_paths_2[index % self.A_size_2]
		
		if self.opt.serial_batches:
			index_B = index % self.B_size
		else:
			index_B = random.randint(0, self.B_size - 1)
		
		B_path = self.B_paths[index_B]
		
		A_label_path_1 = self.A_labels_1[index % self.A_size_1]
		A_label_path_2 = self.A_labels_2[index % self.A_size_2]
		
		A_label_1 = Image.open(A_label_path_1)
		A_label_2 = Image.open(A_label_path_2)
		
		# remaping label for synthia
		A_label_1 = np.asarray(A_label_1)
		A_label_1 = syn_relabel(A_label_1)
		A_label_1 = Image.fromarray(A_label_1, 'L')
		
		# remaping label for gta5
		
		A_label_2 = np.asarray(A_label_2)
		A_label_2 = remap_labels_to_train_ids(A_label_2)
		A_label_2 = Image.fromarray(A_label_2, 'L')
		
		A_img_1 = Image.open(A_path_1).convert('RGB')
		A_img_2 = Image.open(A_path_2).convert('RGB')
		
		B_img = Image.open(B_path).convert('RGB')
		
		A_1 = self.transform(A_img_1)
		A_2 = self.transform(A_img_2)
		
		B = self.transform(B_img)
		
		A_label_1 = self.label_transform(A_label_1)
		A_label_2 = self.label_transform(A_label_2)
		
		if self.opt.which_direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc
		
		if input_nc == 1:  # RGB to gray
			tmp = A_1[0, ...] * 0.299 + A_1[1, ...] * 0.587 + A_1[2, ...] * 0.114
			A_1 = tmp.unsqueeze(0)
			
			tmp = A_2[0, ...] * 0.299 + A_2[1, ...] * 0.587 + A_2[2, ...] * 0.114
			A_2 = tmp.unsqueeze(0)
		
		if output_nc == 1:  # RGB to gray
			tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
			B = tmp.unsqueeze(0)
		
		return {'A_1': A_1, 'A_2': A_2, 'B': B, 'A_paths_1': A_path_1, 'A_paths_2': A_path_2, 'B_paths': B_path, 'A_label_1': A_label_1,
		        'A_label_2': A_label_2}
	
	def __len__(self):
		return max(self.A_size_1, self.B_size, self.A_size_2)
	
	def name(self):
		return 'GTA5_Synthia_Cityscapes'
