import logging
import os.path
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


def supervised_loss(score, label, weights=None):
	loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True, ignore_index=255)
	loss = loss_fn_(F.log_softmax(score), label)
	return loss


@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--resize_to', type=int, default=720)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--adam/--sgd', default=False)
@click.option('--small', type=int, default=2)
@click.option('--preprocessing', default=False)
@click.option('--force_split', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model_weights', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=19, type=int)
@click.option('--nthreads', default=8, type=int)
@click.option('--gpu', default='0')
@click.option('--start_step', default=0)
@click.option('--data_flag', default='', type=str)
@click.option('--rundir_flag', default='', type=str)
@click.option('--serial_batches', type=bool, default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
def main(output, dataset, datadir, batch_size, lr, step, iterations,
         momentum, snapshot, downscale, augmentation, fyu, crop_size,
         weights, model, gpu, num_cls, nthreads, model_weights, data_flag,
         serial_batches, resize_to, start_step, preprocessing, small, rundir_flag, force_split, adam):
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
		net = get_model(model, num_cls=num_cls, weights_init=model_weights)
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
		transform.extend([torchvision.transforms.Resize([int(resize_to), int(int(resize_to) * 1.8)])])
		target_transform.extend([torchvision.transforms.Resize([int(resize_to), int(int(resize_to) * 1.8)], interpolation=Image.NEAREST)])
	
	transform.extend([net.module.transform])
	target_transform.extend([to_tensor_raw])
	transform = torchvision.transforms.Compose(transform)
	target_transform = torchvision.transforms.Compose(target_transform)
	
	if force_split:
		datasets = []
		datasets.append(
			get_dataset(dataset[0], os.path.join(datadir, dataset[0]), num_cls=num_cls, transform=transform, target_transform=target_transform,
			            data_flag=data_flag))
		datasets.append(
			get_dataset(dataset[1], os.path.join(datadir, dataset[1]), num_cls=num_cls, transform=transform, target_transform=target_transform))
	else:
		datasets = [get_dataset(name, os.path.join(datadir, name), num_cls=num_cls, transform=transform, target_transform=target_transform,
		                        data_flag=data_flag) for name in dataset]
	
	if weights is not None:
		weights = np.loadtxt(weights)
	
	if adam:
		print("Using Adam")
		opt = torch.optim.Adam(net.module.parameters(), lr=1e-4)
	else:
		print("Using SGD")
		opt = torch.optim.SGD(net.module.parameters(), lr=lr, momentum=momentum, weight_decay=0.0005)
	
	if augmentation:
		collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
	else:
		collate_fn = torch.utils.data.dataloader.default_collate
	
	loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=not serial_batches, num_workers=nthreads, collate_fn=collate_fn,
	                                       pin_memory=True) for dataset in datasets]
	iteration = start_step
	losses = deque(maxlen=10)
	
	for loader in loaders:
		loader.dataset.__getitem__(0, debug=True)
	
	for im, label in roundrobin_infinite(*loaders):
		# Clear out gradients
		opt.zero_grad()
		
		# load data/label
		im = make_variable(im, requires_grad=False)
		label = make_variable(label, requires_grad=False)
		
		if iteration == 0:
			print("im size: {}".format(im.size()))
			print("label size: {}".format(label.size()))
		
		# forward pass and compute loss
		preds = net(im)
		loss = supervised_loss(preds, label)
		
		# backward pass
		loss.backward()
		losses.append(loss.item())
		
		# step gradients
		opt.step()
		
		# log results
		if iteration % 10 == 0:
			logging.info('Iteration {}:\t{}'.format(iteration, np.mean(losses)))
			writer.add_scalar('loss', np.mean(losses), iteration)
		iteration += 1
		if step is not None and iteration % step == 0:
			logging.info('Decreasing learning rate by 0.1.')
			step_lr(opt, 0.1)
		
		if iteration % snapshot == 0:
			torch.save(net.module.state_dict(),
			           '{}/iter_{}.pth'.format(output, iteration))
		
		if iteration >= iterations:
			logging.info('Optimization complete.')


if __name__ == '__main__':
	main()
