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

def display_imgs(train_loader):
	images, labels = next(iter(train_loader))
	fig=plt.figure(figsize=(20,8))
	for i in range(20):
		ax=fig.add_subplot(2,10, i+1)
		img=np.squeeze(images[i].numpy())
		img=img/2 +0.5
		img=np.transpose(img, (1, 2, 0))
		ax.imshow(img)
		#ax.set_title(str(classes[labels[i].item()]))

