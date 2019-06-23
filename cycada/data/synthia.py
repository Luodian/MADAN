import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image

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

id2label_16 = {0: ignore_label,
               1: 9,
               2: 2,
               3: 0,
               4: 1,
               5: 4,
               6: 8,
               7: 5,
               8: 12,
               9: 7,
               10: 10,
               11: 15,
               12: 14,
               13: ignore_label,
               14: ignore_label,
               15: 6,
               16: ignore_label,
               17: 11,
               18: ignore_label,
               19: 13,
               20: ignore_label,
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


def syn_relabel_16(arr):
	out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
	for id, label in id2label_16.items():
		out[arr == id] = int(label)
	return out


@register_data_params('synthia')
class SYNTHIAParams(DatasetParams):
	num_channels = 3
	image_size = 1024
	mean = 0.5
	std = 0.5
	num_cls = 19
	target_transform = None


@register_dataset_obj('synthia')
class SYNTHIA(data.Dataset):
	
	def __init__(self, root, num_cls=19, split='train', remap_labels=True, transform=None, target_transform=None, data_flag=None, small=2):
		self.root = root
		self.split = split
		self.small = small
		self.remap_labels = remap_labels
		self.ids = self.collect_ids()
		self.transform = transform
		self.target_transform = target_transform
		self.classes = classes
		self.num_cls = num_cls
		self.data_flag = data_flag
	
	def collect_ids(self):
		splits = []
		with open(os.path.join(self.root, 'SYNTHIA_imagelist_{}.txt'.format(self.split))) as f:
			for line in f:
				line = line.strip('\n')
				splits.append(line.split('/')[-1])
		return splits
	
	def img_path(self, filename):
		if self.small == 0:
			return os.path.join(self.root, 'RGB_300x540', filename)
		elif self.small == 1:
			return os.path.join(self.root, 'RGB_600x1080', filename)
		else:
			return os.path.join(self.root, 'RGB', filename)
	
	def label_path(self, filename):
		if self.small == 0:
			return os.path.join(self.root, 'GT', 'parsed_LABELS_300x540', filename)
		elif self.small == 1:
			return os.path.join(self.root, 'GT', 'parsed_LABELS_600x1080', filename)
		else:
			return os.path.join(self.root, 'GT', 'parsed_LABELS', filename)
	
	def __getitem__(self, index, debug=False):
		id = self.ids[index]
		img_path = self.img_path(id)
		label_path = self.label_path(id)
		
		if debug:
			print(self.__class__.__name__)
			print("IMG Path: {}".format(img_path))
			print("Label Path: {}".format(label_path))
		
		img = Image.open(img_path).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		target = Image.open(label_path)
		
		if self.remap_labels:
			target = np.asarray(target)
			if self.num_cls == 16:
				target = syn_relabel_16(target)
			else:
				target = syn_relabel(target)
			target = Image.fromarray(target, 'L')
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		# if img.size()[1] == target.size()[0] and img.size()[1] == 760 and img.size()[2] == target.size()[1] and img.size()[2] == 1280:
		# 	pass
		# else:
		# 	print(img_path)
		# 	print(label_path)
		return img, target
	
	def __len__(self):
		return len(self.ids)
