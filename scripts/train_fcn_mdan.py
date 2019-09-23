import itertools
import json
import logging
import os.path
import subprocess
import sys
from collections import deque

import click
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter

sys.path.append('/nfs/project/libo_iMADAN')

from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.models.MDAN import MDANet
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw, step_lr
from cycada.tools.util import make_variable


def to_tensor_raw(im):
	return torch.from_numpy(np.array(im, np.int64, copy=False))


def roundrobin_infinite(*loaders):
	if not loaders:
		return
	iters = [iter(loader) for loader in loaders]
	while True:
		for i in range(len(iters)):
			it = iters[i]
			try:
				yield next(it)
			except StopIteration:
				iters[i] = iter(loaders[i])
				yield next(iters[i])


def multi_source_infinite(loaders, target_loader):
	if not loaders:
		return
	iters_syn = iter(loaders[0])
	iters_gta = iter(loaders[1])
	iters_cs = iter(target_loader)
	
	while True:
		try:
			yield next(iters_syn), next(iters_gta), next(iters_cs)
		except StopIteration:
			iters_syn = iter(loaders[0])
			iters_gta = iter(loaders[1])
			iters_cs = iter(target_loader)
			yield next(iters_syn), next(iters_gta), next(iters_cs)


def supervised_loss(score, label, weights=None):
	loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True, ignore_index=255)
	loss = loss_fn_(F.log_softmax(score), label)
	return loss

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--target_name', required=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--resize_to', type=int, default=720)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--small', type=int, default=2)
@click.option('--preprocessing', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model_weights', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=19, type=int)
@click.option('--nthreads', default=16, type=int)
@click.option('--gpu', default='0')
@click.option('--start_step', default=0)
@click.option('--data_flag', default='', type=str)
@click.option('--rundir_flag', default='', type=str)
@click.option('--serial_batches', type=bool, default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
def main(output, dataset, target_name, datadir, batch_size, lr, iterations,
         momentum, snapshot, downscale, augmentation, fyu, crop_size,
         weights, model, gpu, num_cls, nthreads, model_weights, data_flag, serial_batches, resize_to, start_step, preprocessing, small, rundir_flag):
	if weights is not None:
		raise RuntimeError("weights don't work because eric is bad at coding")
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	config_logging()
	logdir_flag = data_flag
	if rundir_flag != "":
		logdir_flag += "_{}".format(rundir_flag)
	
	logdir = 'runs/{:s}/{:s}/{:s}'.format(model, '-'.join(dataset), logdir_flag)
	writer = SummaryWriter(log_dir=logdir)
	if model == 'fcn8s':
		net = get_model(model, num_cls=num_cls, weights_init=model_weights, output_last_ft=True)
	else:
		net = get_model(model, num_cls=num_cls, finetune=True, weights_init=model_weights)
	net.cuda()
	
	str_ids = gpu.split(',')
	gpu_ids = []
	for str_id in str_ids:
		id = int(str_id)
		if id >= 0:
			gpu_ids.append(id)
	
	# set gpu ids
	if len(gpu_ids) > 0:
		torch.cuda.set_device(gpu_ids[0])
		assert (torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)
	
	transform = []
	target_transform = []
	
	if preprocessing:
		transform.extend([torchvision.transforms.Resize([int(resize_to), int(int(resize_to) * 1.8)], interpolation=Image.BICUBIC)])
		target_transform.extend([torchvision.transforms.Resize([int(resize_to), int(int(resize_to) * 1.8)], interpolation=Image.NEAREST)])
	
	transform.extend([net.module.transform])
	target_transform.extend([to_tensor_raw])
	transform = torchvision.transforms.Compose(transform)
	target_transform = torchvision.transforms.Compose(target_transform)
	
	datasets = [get_dataset(name, os.path.join(datadir, name), num_cls=num_cls, transform=transform, target_transform=target_transform,
	                        data_flag=data_flag, small=small) for name in dataset]
	
	target_dataset = get_dataset(target_name, os.path.join(datadir, target_name), num_cls=num_cls, transform=transform,
	                             target_transform=target_transform,
	                             data_flag=data_flag, small=small)
	
	if weights is not None:
		weights = np.loadtxt(weights)
	
	if augmentation:
		collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
	else:
		collate_fn = torch.utils.data.dataloader.default_collate
	
	loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=not serial_batches, num_workers=nthreads, collate_fn=collate_fn,
	                                       pin_memory=True, drop_last=True) for dataset in datasets]
	
	target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=not serial_batches, num_workers=nthreads,
	                                            collate_fn=collate_fn,
	                                            pin_memory=True, drop_last=True)
	iteration = start_step
	losses = deque(maxlen=10)
	losses_domain_syn = deque(maxlen=10)
	losses_domain_gta = deque(maxlen=10)
	losses_task = deque(maxlen=10)
	
	for loader in loaders:
		loader.dataset.__getitem__(0, debug=True)
	
	input_dim = 2048
	configs = {"input_dim": input_dim, "hidden_layers": [1000, 500, 100], "num_classes": 2, 'num_domains': 2, 'mode': 'dynamic', 'mu': 1e-2,
	           'gamma': 10.0}
	
	mdan = MDANet(configs).to(gpu_ids[0])
	mdan = torch.nn.DataParallel(mdan, gpu_ids)
	mdan.train()
	
	opt = torch.optim.Adam(itertools.chain(mdan.module.parameters(), net.module.parameters()), lr=1e-4)
	
	# cnt = 0
	for (im_syn, label_syn), (im_gta, label_gta), (im_cs, label_cs) in multi_source_infinite(loaders, target_loader):
		# cnt += 1
		# print(cnt)
		# Clear out gradients
		opt.zero_grad()
		
		# load data/label
		im_syn = make_variable(im_syn, requires_grad=False)
		label_syn = make_variable(label_syn, requires_grad=False)
		
		im_gta = make_variable(im_gta, requires_grad=False)
		label_gta = make_variable(label_gta, requires_grad=False)
		
		im_cs = make_variable(im_cs, requires_grad=False)
		label_cs = make_variable(label_cs, requires_grad=False)
		
		if iteration == 0:
			print("im_syn size: {}".format(im_syn.size()))
			print("label_syn size: {}".format(label_syn.size()))
			
			print("im_gta size: {}".format(im_gta.size()))
			print("label_gta size: {}".format(label_gta.size()))
			
			print("im_cs size: {}".format(im_cs.size()))
			print("label_cs size: {}".format(label_cs.size()))
		
		if not (im_syn.size() == im_gta.size() == im_cs.size()):
			print(im_syn.size())
			print(im_gta.size())
			print(im_cs.size())
		
		# forward pass and compute loss
		preds_syn, ft_syn = net(im_syn)
		# pooled_ft_syn = avg_pool(ft_syn)
		
		preds_gta, ft_gta = net(im_gta)
		# pooled_ft_gta = avg_pool(ft_gta)
		
		preds_cs, ft_cs = net(im_cs)
		# pooled_ft_cs = avg_pool(ft_cs)
		
		loss_synthia = supervised_loss(preds_syn, label_syn)
		loss_gta = supervised_loss(preds_gta, label_gta)
		
		loss = loss_synthia + loss_gta
		losses_task.append(loss.item())
		
		logprobs, sdomains, tdomains = mdan(ft_syn, ft_gta, ft_cs)
		
		slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(gpu_ids[0])
		tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(gpu_ids[0])
		
		# TODO: increase task loss
		# Compute prediction accuracy on multiple training sources.
		domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) + F.nll_loss(tdomains[j], tlabels) for j in range(configs['num_domains'])])
		losses_domain_syn.append(domain_losses[0].item())
		losses_domain_gta.append(domain_losses[1].item())
		
		# Different final loss function depending on different training modes.
		if configs['mode'] == "maxmin":
			loss = torch.max(loss) + configs['mu'] * torch.min(domain_losses)
		elif configs['mode'] == "dynamic":
			loss = torch.log(torch.sum(torch.exp(configs['gamma'] * (loss + configs['mu'] * domain_losses)))) / configs['gamma']
		
		# backward pass
		loss.backward()
		losses.append(loss.item())
		
		torch.nn.utils.clip_grad_norm_(net.module.parameters(), 10)
		torch.nn.utils.clip_grad_norm_(mdan.module.parameters(), 10)
		# step gradients
		opt.step()
		
		# log results
		if iteration % 10 == 0:
			logging.info(
				'Iteration {}:\t{:.3f} Domain SYN: {:.3f} Domain GTA: {:.3f} Task: {:.3f}'.format(iteration, np.mean(losses),
				                                                                                  np.mean(losses_domain_syn),
				                                                                                  np.mean(losses_domain_gta), np.mean(losses_task)))
			writer.add_scalar('loss', np.mean(losses), iteration)
			writer.add_scalar('domain_syn', np.mean(losses_domain_syn), iteration)
			writer.add_scalar('domain_gta', np.mean(losses_domain_gta), iteration)
			writer.add_scalar('task', np.mean(losses_task), iteration)
		iteration += 1
		
		if iteration % 500 == 0:
			os.makedirs(output, exist_ok=True)
			torch.save(net.module.state_dict(), '{}/net-itercurr.pth'.format(output))
		
		if iteration % snapshot == 0:
			torch.save(net.module.state_dict(), '{}/iter_{}.pth'.format(output, iteration))
		
		if iteration >= iterations:
			logging.info('Optimization complete.')


if __name__ == '__main__':
	main()
