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
from utils.data_albument import TrainAlbumentation,TestAlbumentation


	
def download_load():	
	use_cuda = torch.cuda.is_available()

	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)
	SEED=1
	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
		torch.cuda.manual_seed(SEED)
	transform_train = TrainAlbumentation()
	transform = TestAlbumentation()
	#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	#transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	dataloader_args = dict(shuffle=True, batch_size=64, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

	#trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_train)
	#testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
	trainset=torchvision.datasets.ImageFolder('/content/tiny-imagenet-200/combined', transform=transform_train)
	testset= torchvision.datasets.ImageFolder('/content/tiny-imagenet-200/new_test', transform=transform)                                    

	train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
	test_loader= torch.utils.data.DataLoader(testset, **dataloader_args)

	#classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return trainset, testset, train_loader, test_loader
		 
