import os
import sys

import torch
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

from data import CreateDataLoader
import logging

sys.path.append("/nfs/project/libo_i/MADAN")

if __name__ == '__main__':
	opt = TestOptions().parse()
	opt.serial_batches = True  # no shuffle
	opt.no_flip = True  # no flip
	opt.display_id = -1  # no visdom display
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	model = create_model(opt)
	model.setup(opt)
	# create website
	web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
	webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
	# test
	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break
		# check img size
		if i == 0:
			for item in data.items():
				if isinstance(item[1], torch.Tensor):
					logging.info(item[0], item[1].size())
		
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		# remove reductant files when outputing
		if opt.out_all:
			remove_list = []
			for item in visuals:
				if 'fake_B' not in item:
					remove_list.append(item)
			
			for rm_item in remove_list:
				del visuals[rm_item]
		
		img_path = model.get_image_paths()
		if i % 5 == 0:
			logging.info('processing (%04d)-th image...' % (i * opt.batchSize))
		if 'mul' in opt.model:
			save_images(webpage.get_image_dir(), visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, multi_flag=True)
		else:
			save_images(webpage.get_image_dir(), visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

