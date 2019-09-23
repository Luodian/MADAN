import os.path

import numpy as np
import torch.utils.data as data
from PIL import Image
from .util import classes, ignore_label, id2label
from .data_loader import register_dataset_obj

@register_dataset_obj('bdds')
class BDDS(data.Dataset):
	def __init__(self, root, num_cls=19, split='train', remap_labels=True, transform=None, target_transform=None, data_flag=None):
		self.root = root
		self.split = split
		self.remap_labels = remap_labels
		self.transform = transform
		self.target_transform = target_transform
		self.classes = classes
		self.data_flag = data_flag
		self.num_cls = num_cls
		self.ids = self.collect_ids()
	
	def collect_ids(self):
		splits = []
		path = os.path.join(self.root, "images", self.split)
		files = os.listdir(path)
		for item in files:
			fip = os.path.join(path, item)
			splits.append(fip.split('/')[-1])
		
		return splits
	
	def img_path(self, filename):
		return os.path.join(self.root, "images", self.split, filename)
	
	def label_path(self, filename):
		return os.path.join(self.root, 'labels', self.split, "{}_train_id.png".format(filename[:-4]))
	
	def __getitem__(self, index, debug=False):
		id = self.ids[index]
		img_path = self.img_path(id)
		label_path = self.label_path(id)
		
		img = Image.open(img_path).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		target = Image.open(label_path)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target
	
	def __len__(self):
		return len(self.ids)
