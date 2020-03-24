import albumentations
from albumentations import *
from albumentations import Compose
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():
	def __init__(self):
		self.train_trans = Compose([
		  HorizontalFlip(),
		Rotate((-10.0, 10.0)),
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
