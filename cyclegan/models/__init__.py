import logging

def create_model(opt):
	model = None
	if opt.model == 'cycle_gan':
		# assert(opt.dataset_mode == 'unaligned')
		from .cycle_gan_model import CycleGANModel
		model = CycleGANModel()
	elif opt.model == 'test':
		from .test_model import TestModel
		model = TestModel()
	elif opt.model == 'multi_cycle_gan_semantic':
		from .multi_cycle_gan_semantic_model import CycleGANSemanticModel
		model = CycleGANSemanticModel()
	elif opt.model == 'cycle_gan_semantic_fcn':
		from .cycle_gan_semantic_model import CycleGANSemanticModel
		model = CycleGANSemanticModel()
	else:
		raise NotImplementedError('model [%s] not implemented.' % opt.model)
	model.initialize(opt)
	logging.info("model [%s] was created" % (model.name()))
	return model
