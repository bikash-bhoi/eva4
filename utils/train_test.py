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
import torch


from tqdm import tqdm


def train(model, device, train_loader, optimizer, epoch,criterion= nn.NLLLoss()):
	train_losses = []
	train_acc = []

	model.train()
	pbar = tqdm(train_loader)
	correct = 0
	processed = 0
	criterion=criterion.to(device)
	
	for batch_idx, (data, target) in enumerate(pbar):
		# get samples
		data, target = data.to(device), target.to(device)

		# Init
		optimizer.zero_grad()
		# In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
		# Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

		# Predict
		y_pred = model(data)

		# Calculate loss
		
		loss  = criterion(y_pred, target)
		

		train_losses.append(loss)

	   

		 #Backpropagation
		loss.backward()
		optimizer.step()

		# Update pbar-tqdm
		
		pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		correct += pred.eq(target.view_as(pred)).sum().item()
		processed += len(data)

		pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
	train_acc.append(100*correct/processed)
	return train_acc[-1]
	

def test(model, device, test_loader,criterion = F.nll_loss):
	test_losses_l1 = []
	test_acc_l1 = []
	model.eval()
	test_loss = 0
	correct = 0
	
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			

			test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)
	test_losses_l1.append(test_loss)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	
	test_acc_l1.append(100. * correct / len(test_loader.dataset))
	return test_acc_l1[-1]

def fit(model, device, train_loader, test_loader, optimizer, scheduler, num_epoch):
	
	for epoch in range(1,num_epoch+1):
		curr_lr=optimizer.param_groups[0]['lr']
		print(f'Epoch: {epoch} Learning_Rate {curr_lr}')
		train_acc1 = train(model, device, train_loader, optimizer, epoch)
		test_acc1 = test(model, device, test_loader)
		#print('Test accuracy:', test_acc1)
		if "ReduceLROnPlateau" in  str(type(scheduler)):
			scheduler.step(test_acc1)
		elif "OneCycleLR" in  str(type(scheduler)):
			scheduler.step()
	
	
def predict(model, device, test_loader):
	pred_all=[]
	model.eval()

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			#incorrect_pred.append(pred.eq(target.view_as(pred)))
			pred_all +=list(pred.squeeze().cpu().numpy())

	return pred_all

def get_misclassified(pred,labels):
	misclassified = []
	correct = []
	for i in (range(len(pred))):
		if pred[i] != labels[i] : misclassified.append((i,pred[i],labels[i]))
		else : correct.append((i,pred[i],labels[i]))
	return correct,	misclassified

	
