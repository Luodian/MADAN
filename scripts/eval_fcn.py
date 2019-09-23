import os
import sys

from torchvision.transforms import transforms

sys.path.append('/nfs/project/libo_iMADAN')
import json
import click
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import *

from cycada.data.data_loader import dataset_obj, get_fcn_dataset
from cycada.models.models import get_model, models
from cycada.util import to_tensor_raw
import torchvision
from PIL import Image

loader = transforms.Compose([
	transforms.ToTensor()])

unloader = transforms.ToPILImage()


def fmt_array(arr, fmt=','):
	strs = ['{:.3f}'.format(x) for x in arr]
	return fmt.join(strs)


def fast_hist(a, b, n):
	k = (a >= 0) & (a < n)
	return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def result_stats(hist):
	acc_overall = np.diag(hist).sum() / hist.sum() * 100
	acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8) * 100
	iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
	freq = hist.sum(1) / hist.sum()
	fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
	return acc_overall, acc_percls, iu, fwIU


@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--dataset', default='cityscapes',
              type=click.Choice(dataset_obj.keys()))
@click.option('--datadir', default='',
              type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--gpu', default='0')
@click.option('--num_cls', default=19)
@click.option('--batch_size', default=16)
@click.option('--loadSize', default=None)
@click.option('--fineSize', default=None)
def main(path, dataset, datadir, model, gpu, num_cls, batch_size, loadSize, fineSize):
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	
	net = get_model(model, num_cls=num_cls, weights_init=path)
	
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
	
	net.eval()
	
	if (loadSize and fineSize) is not None:
		print("Loading Center Crop DataLoader Transform")
		data_transform = torchvision.transforms.Compose([transforms.Resize([int(loadSize), int(int(fineSize) * 1.8)], interpolation=Image.BICUBIC),
		                                                 net.module.transform.transforms[0], net.module.transform.transforms[1]])
		
		target_transform = torchvision.transforms.Compose([transforms.Resize([int(loadSize), int(int(fineSize) * 1.8)], interpolation=Image.NEAREST),
			 transforms.Lambda(lambda img: to_tensor_raw(img))])
	
	else:
		data_transform = net.module.transform
		target_transform = torchvision.transforms.Compose([transforms.Lambda(lambda img: to_tensor_raw(img))])
	
	ds = get_fcn_dataset(dataset, datadir, num_cls=num_cls, split='val', transform=data_transform, target_transform=target_transform)
	classes = ds.classes
	
	loader = torch.utils.data.DataLoader(ds, num_workers=16, batch_size=batch_size)

	errs = []
	hist = np.zeros((num_cls, num_cls))
	if len(loader) == 0:
		print('Empty data loader')
		return
	iterations = tqdm(enumerate(loader))
	for im_i, (im, label) in iterations:
		if im_i == 0:
			print(im.size())
			print(label.size())
		
		if im_i > 32:
			break
		
		im = Variable(im.cuda())
		score = net(im).data
		_, preds = torch.max(score, 1)
		hist += fast_hist(label.numpy().flatten(), preds.cpu().numpy().flatten(), num_cls)
		acc_overall, acc_percls, iu, fwIU = result_stats(hist)
		iterations.set_postfix({'mIoU': ' {:0.2f}  fwIoU: {:0.2f} pixel acc: {:0.2f} per cls acc: {:0.2f}'.format(np.nanmean(iu), fwIU, acc_overall,
		                                                                                                          np.nanmean(acc_percls))})
	print()
	
	synthia_metric_iu = 0
	
	# line = ""
	for index, item in enumerate(classes):
		print(classes[index], " {:0.1f}".format(iu[index]))
		if classes[index] != 'terrain' and classes[index] != 'truck' and classes[index] != 'train':
			synthia_metric_iu += iu[index]
			# line += " {:0.1f} &".format(iu[index])
			
	# variable "line" is used for adding format results into latex grids
	# print(line)
	
	print(np.nanmean(iu), fwIU, acc_overall, np.nanmean(acc_percls))
	print("16 Class-Wise mIOU is {}".format(synthia_metric_iu / 16))
	print('Errors:', errs)
	
	cur_path = path.split('/')[-1]
	parent_path = path.replace(cur_path, '')
	results_dict_path = os.path.join(parent_path, 'result.json')
	results_dict = {}
	results_dict[cur_path] = [np.nanmean(iu), synthia_metric_iu / 16]
	
	if os.path.exists(results_dict_path) is False:
		with open(results_dict_path, 'w') as fp:
			json.dump(results_dict, fp)
	else:
		with open(results_dict_path, 'r') as fp:
			exist_dict = json.load(fp)
		
		with open(results_dict_path, 'w') as fp:
			exist_dict.update(results_dict)
			json.dump(exist_dict, fp)


if __name__ == '__main__':
	main()
