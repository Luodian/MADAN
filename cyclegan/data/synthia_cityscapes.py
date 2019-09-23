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


class SynthiaCityscapesDataset(BaseDataset):
	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.dir_A = os.path.join(opt.dataroot, 'synthia', 'RGB')
		self.dir_B = os.path.join(opt.dataroot, 'cityscapes', 'leftImg8bit')
		self.dir_A_label = os.path.join(opt.dataroot, 'synthia', 'GT', 'parsed_LABELS')
		self.dir_B_label = os.path.join(opt.dataroot, 'cityscapes', 'gtFine')
		
		self.A_paths = make_dataset(self.dir_A)
		self.B_paths = make_dataset(self.dir_B)
		
		self.A_paths = sorted(self.A_paths)
		self.B_paths = sorted(self.B_paths)
		self.A_size = len(self.A_paths)
		self.B_size = len(self.B_paths)
		
		self.A_labels = make_dataset(self.dir_A_label)
		self.B_labels = make_cs_labels(self.dir_B_label)
		
		self.A_labels = sorted(self.A_labels)
		self.B_labels = sorted(self.B_labels)
		
		self.transform = get_transform(opt)
		self.label_transform = get_label_transform(opt)
	
	def __getitem__(self, index):
		A_path = self.A_paths[index % self.A_size]
		if self.opt.serial_batches:
			index_B = index % self.B_size
		else:
			index_B = random.randint(0, self.B_size - 1)
		B_path = self.B_paths[index_B]
		
		A_label_path = self.A_labels[index % self.A_size]
		B_label_path = self.B_labels[index_B]
		
		A_label = Image.open(A_label_path)
		B_label = Image.open(B_label_path)
		
		A_label = np.asarray(A_label)
		A_label = syn_relabel(A_label)
		
		A_label = Image.fromarray(A_label, 'L')
		B_label = np.asarray(B_label)
		B_label = remap_labels_to_train_ids(B_label)
		B_label = Image.fromarray(B_label, 'L')
		
		A_img = Image.open(A_path).convert('RGB')
		B_img = Image.open(B_path).convert('RGB')
		
		A = self.transform(A_img)
		B = self.transform(B_img)
		
		A_label = self.label_transform(A_label)
		B_label = self.label_transform(B_label)
		
		if self.opt.which_direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc
		
		if input_nc == 1:  # RGB to gray
			tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
			A = tmp.unsqueeze(0)
		
		if output_nc == 1:  # RGB to gray
			tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
			B = tmp.unsqueeze(0)
		return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_label': A_label, 'B_label': B_label}
	
	def __len__(self):
		return max(self.A_size, self.B_size)
	
	def name(self):
		return 'Synthia_Cityscapes'
