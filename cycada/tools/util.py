from functools import partial

import torch
from torch.autograd import Variable


def make_variable(tensor, volatile=False, requires_grad=True):
	if torch.cuda.is_available():
		tensor = tensor.cuda()
	if volatile:
		requires_grad = False
	return Variable(tensor, volatile=volatile, requires_grad=requires_grad)


def pairwise_distance(x, y):
	if not len(x.shape) == len(y.shape):
		raise ValueError('Both inputs should be matrices.')
	
	if x.shape[1] != y.shape[1]:
		raise ValueError('The number of features should be the same.')
	
	x = x.view(x.shape[0], x.shape[1], 1)
	y = torch.transpose(y, 0, 1)
	output = torch.sum((x - y) ** 2, 1)
	output = torch.transpose(output, 0, 1)
	
	return output


def gaussian_kernel_matrix(x, y, sigmas):
	sigmas = sigmas.view(sigmas.shape[0], 1)
	beta = 1. / (2. * sigmas)
	dist = pairwise_distance(x, y).contiguous()
	dist_ = dist.view(1, -1)
	s = torch.matmul(beta, dist_)
	
	return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
	cost = torch.mean(kernel(x, x))
	cost += torch.mean(kernel(y, y))
	cost -= 2 * torch.mean(kernel(x, y))
	
	return cost


def mmd_loss(source_features, target_features):
	sigmas = [
		1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
		1e3, 1e4, 1e5, 1e6
	]
	gaussian_kernel = partial(
		gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
	)
	loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
	loss_value = loss_value
	
	return loss_value
