import torchvision
import torchsummary
from torchsummary import summary

def disp_summary(model,img_size = [3,32,32]):
	#use_cuda= torch.cuda.is_available()
	#device=torch.device('cuda' if use_cuda else 'cpu')
	##model=Net().to(device)
	summary(model, input_size=(img_size[0],img_size[1],img_size[2]))
