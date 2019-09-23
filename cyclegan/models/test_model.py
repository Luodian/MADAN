from . import networks
from .base_model import BaseModel


class TestModel(BaseModel):
	def name(self):
		return 'TestModel'
	
	def initialize(self, opt):
		assert (not opt.isTrain)
		BaseModel.initialize(self, opt)
		
		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = []
		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		self.visual_names = ['real_A']
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		
		if opt.dataset_mode == 'synthia_cityscapes':
			self.model_names = ['G_A_1']
			self.visual_names.append('fake_B_1')
			self.netG_A_1 = networks.define_G(opt.input_nc, opt.output_nc,
			                                  opt.ngf, opt.which_model_netG,
			                                  opt.norm, not opt.no_dropout,
			                                  opt.init_type,
			                                  self.gpu_ids)
		
		elif opt.dataset_mode == 'gta5_cityscapes':
			self.model_names = ['G_A_2']
			self.visual_names.append('fake_B_2')
			self.netG_A_2 = networks.define_G(opt.input_nc, opt.output_nc,
			                                  opt.ngf, opt.which_model_netG,
			                                  opt.norm, not opt.no_dropout,
			                                  opt.init_type,
			                                  self.gpu_ids)
	
	def set_input(self, input):
		# we need to use single_dataset mode
		self.real_A = input['A'].to(self.device)
		self.image_paths = input['A_paths']
	
	def forward(self):
		if hasattr(self, 'netG_A_1'):
			self.fake_B_1 = self.netG_A_1(self.real_A)
		elif hasattr(self, 'netG_A_2'):
			self.fake_B_2 = self.netG_A_2(self.real_A)
