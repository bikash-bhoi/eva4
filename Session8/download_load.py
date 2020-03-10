from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary


	
def download_load():	
	use_cuda = torch.cuda.is_available()

	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)
	SEED=1
	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
		torch.cuda.manual_seed(SEED)

	#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_transforms = transforms.Compose([
                         #  transforms.Resize((28, 28)),
                         #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                          transforms.RandomRotation(7),
						  transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                          ])

	# Test Phase transformations
	test_transforms = transforms.Compose([
										  #  transforms.Resize((28, 28)),
										  #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
										   transforms.ToTensor(),
										   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
										   ])

	dataloader_args = dict(shuffle=True, batch_size=64, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

	trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transforms)
	testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transforms)
                                       

	train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	test_loader= torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return trainset, testset, train_loader, test_loader, classes
		 
