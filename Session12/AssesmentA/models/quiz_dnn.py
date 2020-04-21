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

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv_block2= nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.inter_block1=nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1,1),bias=False, padding_mode='same'))
		self.conv_block3= nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.trans_block4=nn.Sequential(nn.MaxPool2d(2,2))
		self.inter_block4=nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1),bias=False, padding_mode='same'))
		self.conv_block5= nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.conv_block6= nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.conv_block7= nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.trans_block8=nn.Sequential(nn.MaxPool2d(2,2))
		self.inter_block8=nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1),bias=False, padding_mode='same'))
		
		self.conv_block9= nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.conv_block10= nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.conv_block11= nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Dropout2d(0.2))
		self.gap12= nn.Sequential(
		nn.AvgPool2d(8))
		self.fc13=nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1,1),bias=False, padding_mode='same')
	
	def forward(self, x1):
		x2=self.conv_block2(x1)
		x1=self.inter_block1(x1)
		x3=self.conv_block3(x1+x2)
		x4=self.trans_block4(x1+x2+x3)
		x5=self.conv_block5(x4)
		x4=self.inter_block4(x4)
		x6=self.conv_block6(x4+x5)
		x7=self.conv_block7(x4+x5+x6)
		x8=self.trans_block4(x5+x6+x7)
		
		x9=self.conv_block9(x8)
		x8=self.inter_block8(x8)
		x10=self.conv_block10(x8+x9)
		x11=self.conv_block11(x8+x9+x10)
		x12=self.gap12(x11)
		x13=self.fc13(x12)
		x13=x13.view(-1,10)
		return F.log_softmax(x13)
	
def disp_summary(model):
	#use_cuda= torch.cuda.is_available()
	#device=torch.device('cuda' if use_cuda else 'cpu')
	##model=Net().to(device)
	summary(model, input_size=(3,32,32))
  
	
