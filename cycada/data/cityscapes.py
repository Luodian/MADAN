import os.path
import sys

import numpy as np
import torch.utils.data as data
from PIL import Image
from .util import classes, ignore_label, id2label
from .data_loader import DatasetParams, register_data_params, register_dataset_obj

def remap_labels_to_train_ids(arr):
	out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
	for id, label in id2label.items():
		out[arr == id] = int(label)
	return out


@register_data_params('cityscapes')
class CityScapesParams(DatasetParams):
	num_channels = 3
	image_size = 1024
	mean = 0.5
	std = 0.5
	num_cls = 19
	target_transform = None


@register_dataset_obj('cityscapes')
class Cityscapes(data.Dataset):
	def __init__(self, root, num_cls=19, split='train', remap_labels=True, transform=None,
	             target_transform=None):
		self.root = root
		sys.path.append(root)
		self.split = split
		self.remap_labels = remap_labels
		self.ids = self.collect_ids()
		self.transform = transform
		self.target_transform = target_transform
		self.num_cls = num_cls
		
		self.id2label = id2label
		self.classes = classes
	
	def collect_ids(self):
		im_dir = os.path.join(self.root, 'leftImg8bit', self.split)
		ids = []
		for dirpath, dirnames, filenames in os.walk(im_dir):
			for filename in filenames:
				if filename.endswith('.png'):
					ids.append('_'.join(filename.split('_')[:3]))
		return ids
	
	def img_path(self, id):
		fmt = 'leftImg8bit/{}/{}/{}_leftImg8bit.png'
		subdir = id.split('_')[0]
		path = fmt.format(self.split, subdir, id)
		return os.path.join(self.root, path)
	
	def label_path(self, id):
		fmt = 'gtFine/{}/{}/{}_gtFine_labelIds.png'
		subdir = id.split('_')[0]
		path = fmt.format(self.split, subdir, id)
		return os.path.join(self.root, path)
	
	def __getitem__(self, index, debug=False):
		id = self.ids[index]
		img = Image.open(self.img_path(id)).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		target = Image.open(self.label_path(id)).convert('L')
		if self.remap_labels:
			target = np.asarray(target)
			target = remap_labels_to_train_ids(target)
			target = Image.fromarray(np.uint8(target), 'L')
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target
	
	def __len__(self):
		return len(self.ids)
