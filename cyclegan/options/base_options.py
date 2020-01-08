import argparse
import os

import torch
from util import util


class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False
	
	def initialize(self):
		self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
		self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
		self.parser.add_argument('--loadSize', type=int, default=600, help='scale images to this size')
		self.parser.add_argument('--fineSize', type=int, default=600, help='then crop to this size')
		self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
		self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
		self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
		self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
		self.parser.add_argument('--which_model_netD', type=str, default='n_layers', help='selects model to use for netD')
		self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
		self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='experiment_name',
		                         help='name of the experiment. It decides where to store samples and models')
		self.parser.add_argument('--dataset_mode', type=str, default='unaligned',
		                         help='chooses how datasets are loaded. [unaligned | aligned | single]')
		self.parser.add_argument('--model', type=str, default='cycle_gan',
		                         help='chooses which model to use. cycle_gan, pix2pix, test')
		self.parser.add_argument('--weights_model_type', type=str, default='drn26',
		                         help='chooses which model to use. drn26, fcn8s')
		self.parser.add_argument('--num_cls', default=19, type=int)
		self.parser.add_argument('--max_epoch', default=20, type=int)
		self.parser.add_argument('--current_epoch', default=0, type=int)
		self.parser.add_argument('--weights_init', type=str)
		self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
		self.parser.add_argument('--serial_batches', action='store_true',
		                         help='if true, takes images in order to make batches, otherwise takes them randomly')
		self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
		self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
		self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
		self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
		self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
		                         help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, '
		                              'only a subset is loaded.')
		self.parser.add_argument('--resize_or_crop', type=str, default='scale_width_and_crop',
		                         help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
		self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
		self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
		self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
		self.parser.add_argument('--suffix', default='', type=str,
		                         help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
		self.parser.add_argument('--out_all', action='store_true', help='output all stylized images(fake_B_{})')
		self.parser.add_argument('--SAD', action='store_true', help='Sub-domain Aggregation Discriminator module')
		self.parser.add_argument('--CCD', action='store_true', help='Cross-domain Cycle Discriminator module')
		self.parser.add_argument('--CCD_weight', type=float, default=1, help='weight for cross domain cycle discriminator loss')
		self.parser.add_argument('--HF_CCD', action='store_true', help='Half Freeze Cross-domain Cycle Discriminator module')
		self.parser.add_argument('--CCD_frozen_epoch', type=int, default=-1)
		self.parser.add_argument('--SAD_frozen_epoch', type=int, default=-1)
		self.parser.add_argument('--Shared_DT', type=bool, default=True, help="Through ")
		self.parser.add_argument('--model_type', type=str, default='fcn8s', help="choose to load which type of model (fcn8s, drn26, deeplabv2)")
		self.parser.add_argument('--semantic_loss', action='store_true', help='use semantic loss')
		self.parser.add_argument('--general_semantic_weight', type=float, default=0.2, help='weight for semantic loss')
		self.parser.add_argument('--weights_syn', type=str, default='', help='init weights for synthia')
		self.parser.add_argument('--weights_gta', type=str, default='', help='init weights for gta')
		
		self.parser.add_argument('--inference_script', type=str, default='', help='inference script')
		self.parser.add_argument('--dynamic_weight', type=float, default=10, help='Weight for Dynamic Semantic Loss(KL div) loss')
		self.initialized = True
	
	def parse(self):
		if not self.initialized:
			self.initialize()
		opt = self.parser.parse_args()
		opt.isTrain = self.isTrain  # train or test
		
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		
		# set gpu ids
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])
		
		args = vars(opt)
		
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		
		if opt.suffix:
			suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
			opt.name = opt.name + suffix
		# save to the disk
		expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		self.opt = opt
		return self.opt
