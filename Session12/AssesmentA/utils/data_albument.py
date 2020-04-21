#pip install albumentations
import albumentations
from albumentations import *
from albumentations import Compose
from albumentations.pytorch import ToTensor
import numpy as np

class TrainAlbumentation():
	def __init__(self):
		self.train_trans = Compose([
		#Resize(40, 40, interpolation= 0, always_apply=True, p=1),
			PadIfNeeded(min_height=72, min_width=72, p=1.0),
			RandomCrop(height=64, width=64, p=1.0),
			HorizontalFlip(p=0.25),
			Rotate(limit=15, p=0.25),
			RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
			CoarseDropout(max_holes=1, max_height=32, max_width=32, min_height=8,
						min_width=8, fill_value=mean*255.0, p=0.5),
		
		Normalize(
			mean = np.array([0.4914, 0.4822, 0.4465]),
			std = np.array([0.2023, 0.1994, 0.2010]),
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
			mean = np.array([0.4914, 0.4822, 0.4465]),
			std = np.array([0.2023, 0.1994, 0.2010]),
		),
		ToTensor()
		])

	def __call__(self, im):
		im = np.array(im)
		im = self.train_trans(image = im)['image']
		return im
