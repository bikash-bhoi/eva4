import albumentations
from albumentations import *
from albumentations import Compose
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():
	def __init__(self):
		self.train_trans = Compose([
		HorizontalFlip(),
		#Rotate((-10.0, 10.0)),
		Normalize(
			mean=[0.485,0.456,0.406],
			std=[0.229,0.224,0.225],
		),
		PadIfNeeded(min_height=40, min_width=40, border_mode=4, always_apply=False,value=0, p=1.0),
		#RandomCrop(height=32, width=32, always_apply=True, p=1.0, padding=4),
		#Cutout(num_holes=8, max_h_size=8, max_w_size=8, always_apply=False, p=0.5),
		ToTensor()
		])

	def __call__(self, im):
		im = np.array(im)
		im = self.train_trans(image = im)['image']
		return im
 
class TestAlbumentation():
	def __init__(self):
		self.train_trans = Compose([
		Normalize(
			mean=[0.485,0.456,0.406],
			std=[0.229,0.224,0.225],
		),
		ToTensor()
		])

	def __call__(self, im):
		im = np.array(im)
		im = self.train_trans(image = im)['image']
		return im
