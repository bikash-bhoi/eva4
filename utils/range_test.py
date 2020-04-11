from utils.train_test import train , test
import matplotlib.pyplot as plt
import torch.optim as optim
from models.custom_resnet import cust_resnet

def lr_range_test(lrs,model,device,train_loader, test_loader):
	train_acc=[]
	test_acc=[]


	for lr in lrs:
		model = cust_resnet().to(device)
		optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0005)
		optimizer.param_groups[0]['lr'] = lr
		print('LR:',optimizer.param_groups[0]['lr'])
		train_acc1 = train(model, device, train_loader, optimizer, 1)
		test_acc1 = test(model, device, test_loader)
		train_acc.append(train_acc1)
		test_acc.append(test_acc1)
	
	
	max_acc = max(train_acc)
	best_lr = lrs[train_acc.index(max_acc)]
	plt.figure(figsize=(10, 8))
	plt.plot(lrs,train_acc)
	ax = plt.gca()
	plt.xlabel('Learning Rate')
	plt.ylabel('Train Accuracy')
	ax.set_xscale('log')
	blr_txt='Peak LR :'+ str(best_lr) +'; Acc :'+ str(max_acc)
	plt.annotate(blr_txt, xy=(best_lr, max_acc), arrowprops=dict(facecolor='black', shrink=0.05))

	plt.show()

	print('LRs used for range  test : ',lrs)
	print('Train Accuracies : ',train_acc)

	print('Test acc: ',max_acc,'LR: ',best_lr)
