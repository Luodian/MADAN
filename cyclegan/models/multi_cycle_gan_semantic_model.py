import itertools
import sys

import torch
import torch.nn.functional as F
from util.image_pool import ImagePool

from . import networks
from .base_model import BaseModel

sys.path.append('/nfs/project/libo_iMADAN')
from cycada.models import get_model


class CycleGANSemanticModel(BaseModel):
	def name(self):
		return 'CycleGANModel'
	
	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		
		self.semantic_loss = opt.semantic_loss
		
		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['D_A_1', 'G_A_1', 'cycle_A_1', 'idt_A_1',
		                   'D_B_1', 'G_B_1', 'cycle_B_1', 'idt_B_1',
		                   'D_A_2', 'G_A_2', 'cycle_A_2', 'idt_A_2',
		                   'D_B_2', 'G_B_2', 'cycle_B_2', 'idt_B_2']
		
		if opt.SAD:
			self.loss_names.extend(['D_3_1', 'G_s1s2'])
		
		if opt.CCD or opt.HF_CCD:
			self.loss_names.extend(['D_21', 'G_s1s21'])
			self.loss_names.extend(['D_12', 'G_s2s12'])
		
		if self.semantic_loss:
			self.loss_names.extend(['sem_syn', 'sem_gta'])
		
		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		visual_names_A_1 = ['real_A_1', 'fake_B_1', 'rec_A_1']
		visual_names_B_1 = ['real_B', 'fake_A_1', 'rec_B_1']
		
		visual_names_A_2 = ['real_A_2', 'fake_B_2', 'rec_A_2']
		visual_names_B_2 = ['fake_A_2', 'rec_B_2']
		
		if self.isTrain and self.opt.lambda_identity > 0.0:
			visual_names_A_1.append('idt_A_1')
			visual_names_B_1.append('idt_B_1')
			
			visual_names_A_2.append('idt_A_2')
			visual_names_B_2.append('idt_B_2')
		
		self.visual_names = visual_names_A_1 + visual_names_B_1 + visual_names_A_2 + visual_names_B_2
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		if self.isTrain:
			# self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
			if opt.Shared_DT:
				self.model_names = ['G_A_1', 'G_B_1', 'D_A', 'D_B_1', 'D_B_2', 'G_A_2', 'G_B_2']
			else:
				self.model_names = ['G_A_1', 'G_B_1', 'D_A_1', 'D_B_1', 'G_A_2', 'G_B_2', 'D_A_2', 'D_B_2']
			if opt.SAD:
				self.model_names.append('D_3')
			
			if opt.CCD or opt.HF_CCD:
				self.model_names.append('D_12')
				self.model_names.append('D_21')
		
		else:  # during test time, only load Gs
			self.model_names = ['G_A_1', 'G_B_1', 'G_A_2', 'G_B_2']
		
		# load/define networks
		# The naming conversion is different from those used in the paper
		# Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
		self.netG_A_1 = networks.define_G(opt.input_nc, opt.output_nc,
		                                  opt.ngf, opt.which_model_netG, opt.norm,
		                                  not opt.no_dropout, opt.init_type, self.gpu_ids)
		self.netG_B_1 = networks.define_G(opt.output_nc, opt.input_nc,
		                                  opt.ngf, opt.which_model_netG, opt.norm,
		                                  not opt.no_dropout, opt.init_type, self.gpu_ids)
		
		self.netG_A_2 = networks.define_G(opt.input_nc, opt.output_nc,
		                                  opt.ngf, opt.which_model_netG, opt.norm,
		                                  not opt.no_dropout, opt.init_type, self.gpu_ids)
		
		self.netG_B_2 = networks.define_G(opt.output_nc, opt.input_nc,
		                                  opt.ngf, opt.which_model_netG, opt.norm,
		                                  not opt.no_dropout, opt.init_type, self.gpu_ids)
		
		if opt.semantic_loss:
			self.netPixelCLS_SYN = get_model(opt.weights_model_type, num_cls=opt.num_cls, pretrained=True, weights_init=opt.weights_syn)
			self.netPixelCLS_GTA = get_model(opt.weights_model_type, num_cls=opt.num_cls, pretrained=True, weights_init=opt.weights_gta)
			if len(self.gpu_ids) > 0:
				assert (torch.cuda.is_available())
				self.netPixelCLS_SYN.to(self.gpu_ids[0])
				self.netPixelCLS_SYN = torch.nn.DataParallel(self.netPixelCLS_SYN, self.gpu_ids)
				self.netPixelCLS_GTA.to(self.gpu_ids[0])
				self.netPixelCLS_GTA = torch.nn.DataParallel(self.netPixelCLS_GTA, self.gpu_ids)
		
		if self.isTrain:
			use_sigmoid = opt.no_lsgan
			if opt.Shared_DT:
				self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
				                                opt.which_model_netD,
				                                opt.n_layers_D, opt.norm, use_sigmoid,
				                                opt.init_type, self.gpu_ids)
			else:
				self.netD_A_1 = networks.define_D(opt.output_nc, opt.ndf,
				                                  opt.which_model_netD,
				                                  opt.n_layers_D, opt.norm, use_sigmoid,
				                                  opt.init_type, self.gpu_ids)
				
				self.netD_A_2 = networks.define_D(opt.output_nc, opt.ndf,
				                                  opt.which_model_netD,
				                                  opt.n_layers_D, opt.norm, use_sigmoid,
				                                  opt.init_type, self.gpu_ids)
			
			self.netD_B_1 = networks.define_D(opt.input_nc, opt.ndf,
			                                  opt.which_model_netD,
			                                  opt.n_layers_D, opt.norm, use_sigmoid,
			                                  opt.init_type, self.gpu_ids)
			
			self.netD_B_2 = networks.define_D(opt.input_nc, opt.ndf,
			                                  opt.which_model_netD,
			                                  opt.n_layers_D, opt.norm, use_sigmoid,
			                                  opt.init_type, self.gpu_ids)
			
			if opt.SAD:
				self.netD_3 = networks.define_D(opt.input_nc, opt.ndf,
				                                opt.which_model_netD,
				                                opt.n_layers_D, opt.norm, use_sigmoid,
				                                opt.init_type, self.gpu_ids)
			if opt.CCD or opt.HF_CCD:
				self.netD_12 = networks.define_D(opt.input_nc, opt.ndf,
				                                 opt.which_model_netD,
				                                 opt.n_layers_D, opt.norm, use_sigmoid,
				                                 opt.init_type, self.gpu_ids)
				self.netD_21 = networks.define_D(opt.input_nc, opt.ndf,
				                                 opt.which_model_netD,
				                                 opt.n_layers_D, opt.norm, use_sigmoid,
				                                 opt.init_type, self.gpu_ids)
		
		if self.isTrain:
			self.fake_A_1_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
			self.fake_B_1_pool = ImagePool(opt.pool_size)
			self.fake_A_2_pool = ImagePool(opt.pool_size)
			self.fake_B_2_pool = ImagePool(opt.pool_size)
			self.fake_A_21_pool = ImagePool(opt.pool_size)
			self.fake_A_12_pool = ImagePool(opt.pool_size)
			# define loss functions
			self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
			self.criterionCycle = torch.nn.L1Loss()
			self.criterionIdt = torch.nn.L1Loss()
			self.criterionSemantic = torch.nn.KLDivLoss(reduction='batchmean')
			# initialize optimizers
			if opt.Shared_DT:
				self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B_1.parameters(),
				                                                    self.netD_B_2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			else:
				self.optimizer_D_1 = torch.optim.Adam(itertools.chain(self.netD_A_1.parameters(), self.netD_B_1.parameters()),
				                                      lr=opt.lr, betas=(opt.beta1, 0.999))
				self.optimizer_D_2 = torch.optim.Adam(itertools.chain(self.netD_A_2.parameters(), self.netD_B_2.parameters()),
				                                      lr=opt.lr, betas=(opt.beta1, 0.999))
			
			self.optimizer_G_1 = torch.optim.Adam(itertools.chain(self.netG_A_1.parameters(), self.netG_B_1.parameters()),
			                                      lr=opt.lr, betas=(opt.beta1, 0.999))
			
			self.optimizer_G_2 = torch.optim.Adam(itertools.chain(self.netG_A_2.parameters(), self.netG_B_2.parameters()),
			                                      lr=opt.lr, betas=(opt.beta1, 0.999))
			
			if opt.SAD:
				self.optimizer_D_3 = torch.optim.Adam(self.netD_3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			
			if opt.CCD or opt.HF_CCD:
				self.optimizer_D_21 = torch.optim.Adam(self.netD_21.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
				self.optimizer_D_12 = torch.optim.Adam(self.netD_12.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
			
			self.optimizers = []
			self.optimizers.append(self.optimizer_G_1)
			self.optimizers.append(self.optimizer_G_2)
			if opt.Shared_DT:
				self.optimizers.append(self.optimizer_D)
			else:
				self.optimizers.append(self.optimizer_D_1)
				self.optimizers.append(self.optimizer_D_2)
			
			if opt.SAD:
				self.optimizers.append(self.optimizer_D_3)
			if opt.CCD or opt.HF_CCD:
				self.optimizers.append(self.optimizer_D_12)
				self.optimizers.append(self.optimizer_D_21)
	
	def set_input(self, input):
		self.real_A_1 = input['A_1'].to(self.device)
		self.real_A_2 = input['A_2'].to(self.device)
		self.real_B = input['B'].to(self.device)
		
		self.image_paths_1 = input['A_paths_1']
		self.image_paths_2 = input['A_paths_2']
		self.image_paths = self.image_paths_1 + self.image_paths_2
		if 'A_label_1' in input and 'B_label' in input and 'A_label_2' in input:
			self.input_A_label_1 = input['A_label_1'].to(self.device)
			self.input_A_label_2 = input['A_label_2'].to(self.device)
			self.input_B_label = input['B_label'].to(self.device)
	
	def forward(self, opt):
		# cycle for data input #1
		self.fake_B_1 = self.netG_A_1(self.real_A_1)
		self.rec_A_1 = self.netG_B_1(self.fake_B_1)
		
		self.fake_A_1 = self.netG_B_1(self.real_B)
		self.rec_B_1 = self.netG_A_1(self.fake_A_1)
		
		# cycle for data input #2
		self.fake_B_2 = self.netG_A_2(self.real_A_2)
		self.rec_A_2 = self.netG_B_2(self.fake_B_2)
		
		self.fake_A_2 = self.netG_B_2(self.real_B)
		self.rec_B_2 = self.netG_A_2(self.fake_A_2)
		
		if opt.CCD:
			# generate s21 for d21 branch
			self.fake_A_21 = self.netG_B_1(self.fake_B_2)
			# generate s12 for d12 branch
			self.fake_A_12 = self.netG_B_2(self.fake_B_1)
		
		if self.isTrain and self.semantic_loss:
			# Forward all four images through classifier
			# Keep predictions from fake images only
			self.pred_real_A_SYN = self.netPixelCLS_SYN(self.real_A_1)
			_, self.gt_pred_A_SYN = self.pred_real_A_SYN.max(1)
			
			self.pred_fake_B_SYN = self.netPixelCLS_SYN(self.fake_B_1)
			_, pfB_SYN = self.pred_fake_B_SYN.max(1)
			
			self.pred_real_A_GTA = self.netPixelCLS_GTA(self.real_A_2)
			_, self.gt_pred_A_GTA = self.pred_real_A_GTA.max(1)
			
			self.pred_fake_B_GTA = self.netPixelCLS_GTA(self.fake_B_2)
			_, pfB_GTA = self.pred_fake_B_GTA.max(1)
	
	def backward_D_basic(self, netD, real, fake, SAD=False):
		# Real
		if SAD == False:
			pred_real = netD(real)
		else:
			pred_real = netD(real.detach())
		
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		# backward
		loss_D.backward()
		return loss_D
	
	def backward_D_A(self, Shared_DT):
		# data 1 A1->B
		fake_B_1 = self.fake_B_1_pool.query(self.fake_B_1)
		if Shared_DT:
			self.loss_D_A_1 = self.backward_D_basic(self.netD_A, self.real_B, fake_B_1)
		else:
			self.loss_D_A_1 = self.backward_D_basic(self.netD_A_1, self.real_B, fake_B_1)
		# data 2 A2->B
		fake_B_2 = self.fake_B_2_pool.query(self.fake_B_2)
		if Shared_DT:
			self.loss_D_A_2 = self.backward_D_basic(self.netD_A, self.real_B, fake_B_2)
		else:
			self.loss_D_A_2 = self.backward_D_basic(self.netD_A_2, self.real_B, fake_B_2)
	
	def backward_D_B(self):
		# data 1 B->A1
		fake_A_1 = self.fake_A_1_pool.query(self.fake_A_1)
		self.loss_D_B_1 = self.backward_D_basic(self.netD_B_1, self.real_A_1, fake_A_1)
		
		# data 2 B->A2
		fake_A_2 = self.fake_A_2_pool.query(self.fake_A_2)
		self.loss_D_B_2 = self.backward_D_basic(self.netD_B_2, self.real_A_2, fake_A_2)
	
	def backward_D(self, which_D):
		if which_D == 'SAD':
			fake_B_1 = self.fake_B_1_pool.query(self.fake_B_1)
			self.loss_D_3_1 = self.backward_D_basic(self.netD_3, self.fake_B_2, fake_B_1, SAD=True)
		
		elif which_D == 'CCD_21':
			fake_A_21 = self.fake_A_21_pool.query(self.fake_A_21)
			self.loss_D_21 = self.backward_D_basic(self.netD_21, self.real_A_1, fake_A_21)
		
		elif which_D == 'CCD_12':
			fake_A_12 = self.fake_A_12_pool.query(self.fake_A_12)
			self.loss_D_12 = self.backward_D_basic(self.netD_12, self.real_A_2, fake_A_12)
		
		else:
			raise Exception("Invalid Choice {}".format(which_D))
	
	# fake_B_2 = self.fake_B_pool.query(self.fake_B_2)
	# self.loss_D_3_2 = self.backward_D_basic(self.netD_3, self.fake_B_1, fake_B_2)
	
	def backward_G(self, opt):
		lambda_idt = self.opt.lambda_identity
		lambda_A = self.opt.lambda_A
		lambda_B = self.opt.lambda_B
		# Identity loss
		if lambda_idt > 0:
			self.idt_A_1 = self.netG_A_1(self.real_B)
			self.loss_idt_A_1 = self.criterionIdt(self.idt_A_1, self.real_B) * lambda_B * lambda_idt
			
			self.idt_A_2 = self.netG_A_2(self.real_B)
			self.loss_idt_A_2 = self.criterionIdt(self.idt_A_2, self.real_B) * lambda_B * lambda_idt
			
			self.idt_B_1 = self.netG_B_1(self.real_A_1)
			self.loss_idt_B_1 = self.criterionIdt(self.idt_B_1, self.real_A_1) * lambda_A * lambda_idt
			
			self.idt_B_2 = self.netG_B_2(self.real_A_2)
			self.loss_idt_B_2 = self.criterionIdt(self.idt_B_2, self.real_A_2) * lambda_A * lambda_idt
		
		else:
			self.loss_idt_A_1 = 0
			self.loss_idt_A_2 = 0
			self.loss_idt_B_1 = 0
			self.loss_idt_B_2 = 0
		
		if opt.Shared_DT:
			self.loss_G_A_1 = 2 * self.criterionGAN(self.netD_A(self.fake_B_1), True)
			self.loss_G_A_2 = 2 * self.criterionGAN(self.netD_A(self.fake_B_2), True)
		else:
			self.loss_G_A_1 = 2 * self.criterionGAN(self.netD_A_1(self.fake_B_1), True)
			self.loss_G_A_2 = 2 * self.criterionGAN(self.netD_A_2(self.fake_B_2), True)
		
		# GAN loss D_B(G_B(B))
		self.loss_G_B_1 = self.criterionGAN(self.netD_B_1(self.fake_A_1), True)
		self.loss_G_B_2 = self.criterionGAN(self.netD_B_2(self.fake_A_2), True)
		
		# Forward cycle loss
		self.loss_cycle_A_1 = self.criterionCycle(self.rec_A_1, self.real_A_1) * lambda_A
		self.loss_cycle_A_2 = self.criterionCycle(self.rec_A_2, self.real_A_2) * lambda_A
		
		# Backward cycle loss
		self.loss_cycle_B_1 = self.criterionCycle(self.rec_B_1, self.real_B) * lambda_B
		self.loss_cycle_B_2 = self.criterionCycle(self.rec_B_2, self.real_B) * lambda_B
		
		# combined loss standard cyclegan
		self.loss_G_1 = self.loss_G_A_1 + self.loss_G_B_1 + self.loss_cycle_A_1 + self.loss_cycle_B_1 + self.loss_idt_A_1 + self.loss_idt_B_1
		self.loss_G_2 = self.loss_G_A_2 + self.loss_G_B_2 + self.loss_cycle_A_2 + self.loss_cycle_B_2 + self.loss_idt_A_2 + self.loss_idt_B_2
		self.loss_G = self.loss_G_1 + self.loss_G_2
		
		if opt.SAD:
			# D3 loss
			if opt.SAD_frozen_epoch != -1 and opt.current_epoch > opt.SAD_frozen_epoch:
				self.loss_G_s1s2 = self.criterionGAN(self.netD_3(self.fake_B_1), True)
			else:
				self.loss_G_s1s2 = 0
			self.loss_G += self.loss_G_s1s2
		
		if opt.CCD:
			# D21 loss
			if opt.CCD_frozen_epoch != -1 and opt.current_epoch > opt.CCD_frozen_epoch:
				self.loss_G_s1s21 = self.criterionGAN(self.netD_21(self.fake_A_21), True)
				self.loss_G += self.loss_G_s1s21 * opt.D1D2_weight
			else:
				self.loss_G_s1s21 = 0
			
			if opt.CCD_frozen_epoch != -1 and opt.current_epoch > opt.CCD_frozen_epoch:
				self.loss_G_s2s12 = self.criterionGAN(self.netD_12(self.fake_A_12), True)
				self.loss_G += self.loss_G_s2s12 * opt.D1D2_weight
			else:
				self.loss_G_s2s12 = 0
		
		if opt.semantic_loss:
			self.loss_sem_syn = opt.dynamic_weight * self.criterionSemantic(F.log_softmax(self.pred_fake_B_SYN, dim=1),
			                                                                F.softmax(self.pred_real_A_SYN, dim=1))
			self.loss_sem_gta = opt.dynamic_weight * self.criterionSemantic(F.log_softmax(self.pred_fake_B_GTA, dim=1),
			                                                                F.softmax(self.pred_real_A_GTA, dim=1))
			self.loss_G += opt.general_semantic_weight * torch.div(self.loss_sem_syn, self.pred_fake_B_SYN.shape[1] * self.pred_fake_B_SYN.shape[2]
			                                                       * self.pred_fake_B_SYN.shape[3])
			self.loss_G += opt.general_semantic_weight * torch.div(self.loss_sem_gta, self.pred_fake_B_GTA.shape[1] * self.pred_fake_B_GTA.shape[2]
			                                                       * self.pred_fake_B_GTA.shape[3])
		
		self.loss_G.backward()
	
	def backward_HF_CCD(self, opt):
		self.fake_B_1 = self.netG_A_1(self.real_A_1)
		self.fake_B_2 = self.netG_A_2(self.real_A_2)
		# generate s21 for d21 branch
		self.fake_A_21 = self.netG_B_1(self.fake_B_2)
		# generate s12 for d12 branch
		self.fake_A_12 = self.netG_B_2(self.fake_B_1)
		
		# D12 loss
		if opt.CCD_frozen_epoch != -1 and opt.current_epoch > opt.CCD_frozen_epoch:
			self.loss_G_s2s12 = self.criterionGAN(self.netD_12(self.fake_A_12), True)
		else:
			self.loss_G_s2s12 = 0
		# D21 loss
		if opt.CCD_frozen_epoch != -1 and opt.current_epoch > opt.CCD_frozen_epoch:
			self.loss_G_s1s21 = self.criterionGAN(self.netD_21(self.fake_A_21), True)
		else:
			self.loss_G_s1s21 = 0
		
		# self.loss_G_s2s12 = self.criterionGAN(self.netD_12(self.fake_A_12), True)
		# self.loss_G_s1s21 = self.criterionGAN(self.netD_21(self.fake_A_21), True)
		self.loss_G_HF = self.loss_G_s1s21 * opt.CCD_weight + self.loss_G_s2s12 * opt.CCD_weight
		
		if isinstance(self.loss_G_HF, torch.Tensor):
			self.loss_G_HF.backward()
	
	def optimize_parameters(self, opt):
		# forward
		self.forward(opt)
		# G_A and G_B
		# set D to false, back prop G's gradients
		if opt.Shared_DT:
			self.set_requires_grad([self.netD_A, self.netD_B_1, self.netD_B_2], False)
		else:
			self.set_requires_grad([self.netD_A_1, self.netD_B_1], False)
			self.set_requires_grad([self.netD_A_2, self.netD_B_2], False)
		
		if opt.SAD:
			self.set_requires_grad([self.netD_3], False)
		
		if opt.CCD or opt.HF_CCD:
			self.set_requires_grad([self.netD_21], False)
			self.set_requires_grad([self.netD_12], False)
		
		self.set_requires_grad([self.netG_A_1, self.netG_B_1], True)
		self.set_requires_grad([self.netG_A_2, self.netG_B_2], True)
		
		self.optimizer_G_1.zero_grad()
		self.optimizer_G_2.zero_grad()
		# self.optimizer_CLS.zero_grad()
		self.backward_G(opt)
		self.optimizer_G_1.step()
		self.optimizer_G_2.step()
		
		if opt.HF_CCD:
			self.optimizer_G_1.zero_grad()
			self.optimizer_G_2.zero_grad()
			self.set_requires_grad([self.netG_A_1, self.netG_A_2], True)
			self.set_requires_grad([self.netG_B_1, self.netG_B_2], False)
			
			self.backward_HF_CCD(opt)
			self.optimizer_G_1.step()
			self.optimizer_G_2.step()
		
		# D_A and D_B
		if opt.Shared_DT:
			self.set_requires_grad([self.netD_A, self.netD_B_1, self.netD_B_2], True)
		else:
			self.set_requires_grad([self.netD_A_1, self.netD_B_1], True)
			self.set_requires_grad([self.netD_A_2, self.netD_B_2], True)
		
		if opt.Shared_DT:
			self.optimizer_D.zero_grad()
		else:
			self.optimizer_D_1.zero_grad()
			self.optimizer_D_2.zero_grad()
		
		self.backward_D_B()
		self.backward_D_A(opt.Shared_DT)
		if opt.Shared_DT:
			self.optimizer_D.step()
		else:
			self.optimizer_D_1.step()
			self.optimizer_D_2.step()
		
		if opt.SAD:
			self.set_requires_grad([self.netD_3], True)
			self.optimizer_D_3.zero_grad()
			self.backward_D('SAD')
			self.optimizer_D_3.step()
		
		if opt.CCD or opt.HF_CCD:
			self.set_requires_grad([self.netD_21], True)
			self.optimizer_D_21.zero_grad()
			self.backward_D('CCD_21')
			self.optimizer_D_21.step()
			
			self.set_requires_grad([self.netD_12], True)
			self.optimizer_D_12.zero_grad()
			self.backward_D('CCD_12')
			self.optimizer_D_12.step()
