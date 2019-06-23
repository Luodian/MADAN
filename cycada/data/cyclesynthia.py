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


@register_data_params('cyclesynthia')
class SYNTHIAParams(DatasetParams):
	num_channels = 3
	image_size = 1024
	mean = 0.5
	std = 0.5
	num_cls = 19
	target_transform = None


@register_dataset_obj('cyclesynthia')
class CycleSYNTHIA(data.Dataset):
	
	def __init__(self, root, num_cls=19, split='train', remap_labels=True, transform=None, target_transform=None):
		self.root = root.replace('cycle', '')
		self.split = split
		self.remap_labels = remap_labels
		self.transform = transform
		self.target_transform = target_transform
		self.classes = classes
		self.num_cls = num_cls
		self.ids = self.collect_ids()
	
	def collect_ids(self):
		splits = []
		if self.data_flag:
			path = os.path.join(self.root, self.data_flag)
		else:
			path = os.path.join(self.root, 'Cycle')
		files = os.listdir(path)
		for item in files:
			fip = os.path.join(path, item)
			if (fip.endswith('_fake_B_1.png') or fip.endswith('_fake_B.png')):
				splits.append(fip.split('/')[-1])
		
		return splits
	
	def img_path(self, filename):
		return os.path.join(self.root, filename)
	
	def label_path(self, filename):
		# Case for loading images generated in multi-source cycle
		# In this case, you will generate fake_B_1 for cyclesynthia dataset and fake_B_2 for cyclegta5
		if filename.endswith('_fake_B_1.png'):
			return os.path.join(self.root, 'GT', 'parsed_LABELS', filename.replace('_fake_B_1.png', '.png'))
		elif filename.endswith('_fake_B.png'):
			return os.path.join(self.root, 'GT', 'parsed_LABELS', filename.replace('_fake_B.png', '.png'))
	
	def __getitem__(self, index, debug=False):
		id = self.ids[index]
		img_path = self.img_path(id)
		label_path = self.label_path(id)
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
		return img, target
	
	def __len__(self):
		return len(self.ids)
