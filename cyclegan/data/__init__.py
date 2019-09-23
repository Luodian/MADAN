import sys

import torch.utils.data
from data.base_data_loader import BaseDataLoader

sys.path.append('/nfs/project/libo_i/MADAN')
from cycada.transforms import augment_collate


def CreateDataLoader(opt):
	data_loader = CustomDatasetDataLoader()
	print(data_loader.name())
	data_loader.initialize(opt)
	return data_loader


def CreateDataset(opt):
	dataset = None
	if opt.dataset_mode == 'synthia_cityscapes':
		from data.synthia_cityscapes import SynthiaCityscapesDataset
		dataset = SynthiaCityscapesDataset()
	elif opt.dataset_mode == 'gta5_cityscapes':
		from data.gta5_cityscapes import GTAVCityscapesDataset
		dataset = GTAVCityscapesDataset()
	elif opt.dataset_mode == 'gta_synthia_cityscapes':
		from data.gta_synthia_cityscapes import GTASynthiaCityscapesDataset
		dataset = GTASynthiaCityscapesDataset()
	elif opt.dataset_mode == 'merged_gta_synthia_cityscapes':
		from data.merged_gta_synthia_cityscapes import MergedGTASynthiaCityscapesDataset
		dataset = MergedGTASynthiaCityscapesDataset()
	else:
		raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
	
	print("dataset [%s] was created" % (dataset.name()))
	dataset.initialize(opt)
	return dataset


class CustomDatasetDataLoader(BaseDataLoader):
	def name(self):
		return 'CustomDatasetDataLoader'
	
	def initialize(self, opt):
		BaseDataLoader.initialize(self, opt)
		self.dataset = CreateDataset(opt)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size=opt.batchSize,
			shuffle=not opt.serial_batches,
			num_workers=int(opt.nThreads))
	
	def load_data(self):
		return self
	
	def __len__(self):
		return min(len(self.dataset), self.opt.max_dataset_size)
	
	def __iter__(self):
		for i, data in enumerate(self.dataloader):
			if i * self.opt.batchSize >= self.opt.max_dataset_size:
				break
			yield data
