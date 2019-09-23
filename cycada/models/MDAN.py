#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
	"""
	Implement the gradient reversal layer for the convenience of domain adaptation neural network.
	The forward part is the identity function while the backward part is the negative function.
	"""
	
	def forward(self, inputs):
		return inputs
	
	def backward(self, grad_output):
		grad_input = grad_output.clone()
		grad_input = -grad_input
		return grad_input


class MDANet(nn.Module):
	"""
	Multi-layer perceptron with adversarial regularizer by domain classification.
	"""
	
	def __init__(self, configs):
		super(MDANet, self).__init__()
		
		self.pooling_layer = nn.AdaptiveAvgPool2d((2, 2))
		self.dim_reduction = nn.Conv2d(4096, 512, kernel_size=1)
		nn.init.xavier_normal_(self.dim_reduction.weight)
		nn.init.constant_(self.dim_reduction.bias, 0.1)
		self.input_dim = configs["input_dim"]
		self.num_hidden_layers = len(configs["hidden_layers"])
		self.num_neurons = [] + [self.input_dim] + configs["hidden_layers"]
		self.num_domains = configs["num_domains"]
		# Parameters of hidden, fully-connected layers, feature learning component.
		self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
		                              for i in range(self.num_hidden_layers)])
		# Parameter of the final softmax classification layer.
		self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
		# Parameter of the domain classification layer, multiple sources single target domain adaptation.
		self.domains = nn.ModuleList([nn.Linear(self.num_neurons[-1], 2) for _ in range(self.num_domains)])
		# Gradient reversal layer.
		self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
	
	def forward(self, sinputs_syn, sinputs_gta, tinputs):
		"""
		:param sinputs:     A list of k inputs from k source domains.
		:param tinputs:     Input from the target domain.
		:return:
		"""
		sinputs_gta = self.pooling_layer(sinputs_gta)
		sinputs_syn = self.pooling_layer(sinputs_syn)
		tinputs = self.pooling_layer(tinputs)
		
		sinputs_gta = self.dim_reduction(sinputs_gta)
		sinputs_syn = self.dim_reduction(sinputs_syn)
		tinputs = self.dim_reduction(tinputs)
		
		b = sinputs_gta.size()[0]
		syn_relu, gta_relu, th_relu = sinputs_syn.view(b, -1), sinputs_gta.view(b, -1), tinputs.view(b, -1)
		assert (syn_relu[0].size()[0] == self.input_dim)
		
		for hidden in self.hiddens:
			syn_relu = F.relu(hidden(syn_relu))
			gta_relu = F.relu(hidden(gta_relu))
		
		for hidden in self.hiddens:
			th_relu = F.relu(hidden(th_relu))
		
		# Classification probabilities on k source domains.
		logprobs = []
		logprobs.append(F.log_softmax(self.softmax(syn_relu), dim=1))
		logprobs.append(F.log_softmax(self.softmax(gta_relu), dim=1))
		
		# Domain classification accuracies.
		sdomains, tdomains = [], []
		sdomains.append(F.log_softmax(self.domains[0](self.grls[0](syn_relu)), dim=1))
		tdomains.append(F.log_softmax(self.domains[0](self.grls[0](th_relu)), dim=1))
		
		sdomains.append(F.log_softmax(self.domains[1](self.grls[1](gta_relu)), dim=1))
		tdomains.append(F.log_softmax(self.domains[1](self.grls[1](th_relu)), dim=1))
		
		return logprobs, sdomains, tdomains
	
	def inference(self, inputs):
		h_relu = inputs
		for hidden in self.hiddens:
			h_relu = F.relu(hidden(h_relu))
		# Classification probability.
		logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
		return logprobs
