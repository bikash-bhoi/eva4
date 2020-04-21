import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR,MultiplicativeLR
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary
import torch


def lr_range_test(model, train, test, train_loader, test_loader):
	device= 'cuda' if torch.cuda.is_available() else 'cpu'
	#model = Net().to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.0001)
	lmbda = lambda epoch: 1.4
	#scheduler = OneCycleLR(optimizer,max_lr=0.5,total_steps=25)
	scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
	learning_lr_trace= []
	for epoch in range(1, 25):
	
		print(f'Epoch: {epoch} Learning_Rate {scheduler.get_lr()}')
		learning_lr_trace.append(scheduler.get_lr())
		train_loss, train_acc=train(model, device, train_loader, optimizer, epoch)
		test_loss, test_acc_l1=test(model, device, test_loader)
		scheduler.step()
	
	return learning_lr_trace, train_acc, test_acc_l1

