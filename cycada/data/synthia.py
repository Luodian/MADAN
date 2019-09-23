import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image
from .util import classes, ignore_label, id2label
from .data_loader import DatasetParams, register_data_params, register_dataset_obj

def syn_relabel(arr):
	out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
	for id, label in id2label.items():
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
			target = syn_relabel(target)
			target = Image.fromarray(target, 'L')
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target
	
	def __len__(self):
		return len(self.ids)
