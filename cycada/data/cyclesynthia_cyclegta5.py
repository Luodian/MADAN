import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image

from .cityscapes import remap_labels_to_train_ids
from .data_loader import DatasetParams, register_data_params, register_dataset_obj

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


@register_data_params('cyclesynthia_cyclegta5')
class SYNTHIAParams(DatasetParams):
	num_channels = 3
	image_size = 1024
	mean = 0.5
	std = 0.5
	num_cls = 19
	target_transform = None

# In this class, we iteratively load transferred images from cyclesynthia and cyclegta5
@register_dataset_obj('cyclesynthia_cyclegta5')
class CycleSYNTHIACycleGTA5(data.Dataset):
	
	def __init__(self, root, num_cls=19, split='train', remap_labels=True, transform=None, target_transform=None):
		self.dataset_name = os.path.basename(root)
		self.parent_path = root.replace(self.dataset_name, '')
		self.syn_name = os.path.join(self.parent_path, 'synthia')
		self.gta_name = os.path.join(self.parent_path, 'cyclegta5')
		self.remap_labels = remap_labels
		self.transform = transform
		self.target_transform = target_transform
		self.classes = classes
		self.num_cls = num_cls
		self.syn_ids = self.collect_ids('syn')
		self.gta_ids = self.collect_ids('gta')
	
	def collect_ids(self, datasets_name):
		splits = []
		if datasets_name == 'syn':
			files = os.listdir(self.syn_name)
			for item in files:
				fip = os.path.join(self.syn_name, item)
				if (fip.endswith('_fake_B_1.png') or fip.endswith('_fake_B.png')):
					splits.append(fip.split('/')[-1])
		
		elif datasets_name == 'gta':
			files = os.listdir(self.gta_name)
			for item in files:
				fip = os.path.join(self.gta_name, item)
				if (fip.endswith('_fake_B_2.png') or fip.endswith('_fake_B.png')):
					splits.append(fip.split('/')[-1])
		
		else:
			print("Don't Recognize {}".format(datasets_name))
		
		return splits
	
	def img_path(self, prefix, filename):
		return os.path.join(prefix, filename)
	
	# Case for loading images generated in multi-source cycle
	# In this case, you will generate fake_B_1 for cyclesynthia dataset and fake_B_2 for cyclegta5
	def syn_label_path(self, filename):
		if filename.endswith('_fake_B_1.png'):
			return os.path.join("/nfs/project/libo_i/MADAN/data/synthia", 'GT', 'parsed_LABELS', filename.replace('_fake_B_1.png', '.png'))
		elif filename.endswith('_fake_B.png'):
			return os.path.join("/nfs/project/libo_i/MADAN/data/synthia", 'GT', 'parsed_LABELS', filename.replace('_fake_B.png', '.png'))
	
	def gta_label_path(self, filename):
		if filename.endswith('_fake_B_2.png'):
			return os.path.join('/nfs/project/libo_i/MADAN/data/cyclegta5', 'labels', filename.replace('_fake_B_2.png', '.png'))
		elif filename.endswith('_fake_B.png'):
			return os.path.join('/nfs/project/libo_i/MADAN/data/cyclegta5', 'labels', filename.replace('_fake_B.png', '.png'))
	
	def __getitem__(self, index, debug=False):
		# we iteratively load images from cyclesynthia and cyclegta5
		if index % 2:
			id = self.syn_ids[index % len(self.syn_ids)]
			img_path = self.img_path(self.syn_name, id)
			label_path = self.syn_label_path(id)
			img = Image.open(img_path).convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
			target = Image.open(label_path)
			if self.remap_labels:
				target = np.asarray(target)
				target = syn_relabel(target)
				target = Image.fromarray(target, 'L')
			if self.target_transform is not None:
				target = self.target_transform(target)
		
		else:
			id = self.gta_ids[index % len(self.gta_ids)]
			img_path = self.img_path(self.gta_name, id)
			label_path = self.gta_label_path(id)
			img = Image.open(img_path).convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
			target = Image.open(label_path)
			if self.remap_labels:
				target = np.asarray(target)
				target = remap_labels_to_train_ids(target)
				target = Image.fromarray(target, 'L')
			if self.target_transform is not None:
				target = self.target_transform(target)
		
		# if debug:
		# 	print(self.__class__.__name__)
		# 	print("IMG Path: {}".format(img_path))
		# 	print("Label Path: {}".format(label_path))
		#
		return img, target
	
	def __len__(self):
		return len(self.syn_ids) + len(self.gta_ids)
