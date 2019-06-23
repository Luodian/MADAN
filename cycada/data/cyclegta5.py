import os.path

import numpy as np
from PIL import Image

from .cityscapes import remap_labels_to_train_ids
from .data_loader import register_dataset_obj
from .gta5 import GTA5  # , LABEL2TRAIN


@register_dataset_obj('cyclegta5')
class CycleGTA5(GTA5):
	def collect_ids(self):
		# ids = GTA5.collect_ids(self)
		existing_ids = []
		if self.data_flag:
			path = os.path.join(self.root, self.data_flag)
		else:
			path = os.path.join(self.root, "images")
		
		files = os.listdir(path)
		for item in files:
			full_path = os.path.join(path, item)
			if os.path.exists(full_path) is False:
				continue
			existing_ids.append(full_path.split('/')[-1])
		return sorted(existing_ids)
	
	def __getitem__(self, index, debug=False):
		filename = self.ids[index]
		if self.data_flag == '' or self.data_flag is None:
			img_path = os.path.join(self.root, "images", filename)
		else:
			img_path = os.path.join(self.root, self.data_flag, filename)
		
		if self.data_flag == '' or self.data_flag is None:
			label_path = os.path.join(self.root, 'labels_600x1080', filename)
		else:
			if filename.endswith('_fake_B.png'):
				label_path = os.path.join(self.root, 'labels_600x1080', filename.replace('_fake_B.png', '.png'))
			elif filename.endswith('_fake_B_2.png'):
				label_path = os.path.join(self.root, 'labels_600x1080', filename.replace('_fake_B_2.png', '.png'))
				
		img = Image.open(img_path).convert('RGB')
		target = Image.open(label_path)
		img = img.resize(target.size, resample=Image.BILINEAR)
		if self.transform is not None:
			img = self.transform(img)
		if self.remap_labels:
			target = np.asarray(target)
			target = remap_labels_to_train_ids(target)
			target = Image.fromarray(target, 'L')
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target
