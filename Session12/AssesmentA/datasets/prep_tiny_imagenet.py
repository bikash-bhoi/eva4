import os
import zipfile
from io import StringIO, BytesIO
import requests
import shutil
from os.path import join
from os import listdir, rmdir
import random
def download_imagenet(url):
	if (os.path.isdir('tiny-imagenet-200')):
		print('Images already present')
		return
	r= requests.get(url, stream=True)
	zip_ref= zipfile.ZipFile(BytesIO(r.content))
	zip_ref.extractall('./')
	zip_ref.close()

def prep_tiny_imagenet():
	download_imagenet('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
	shutil.copytree('/content/tiny-imagenet-200/train', '/content/tiny-imagenet-200/combined')
	d = {}
	with open('/content/tiny-imagenet-200/val/val_annotations.txt','r') as f:
	for row in csv.reader(f, delimiter='\t'):
		d[row[0]]=row[1]

	for name, clas in d.items():
	#print(name, clas)
	full_file_name=f'/content/tiny-imagenet-200/val/images/{name}'
	dest=f'/content/tiny-imagenet-200/combined/{clas}'
	shutil.copy(full_file_name, dest)
	
	#move files from images folder to class folder
	for folder in listdir('/content/tiny-imagenet-200/combined'):
		source= f'/content/tiny-imagenet-200/combined/{folder}/images/'
	for filename in listdir(source):
		file=f'/content/tiny-imagenet-200/combined/{folder}/images/{filename}'
		dest=f'/content/tiny-imagenet-200/combined/{folder}/'
		shutil.move(file, dest)
		
	#remove images folder
	for folder in listdir('/content/tiny-imagenet-200/combined'):
		img_fdl= f'/content/tiny-imagenet-200/combined/{folder}/images'
		shutil.rmtree(img_fdl)
		
	os.mkdir('/content/tiny-imagenet-200/new_test') 
	
	#create new test

	random.seed(0)
	for folder in listdir('/content/tiny-imagenet-200/combined'):
		lis_file= os.listdir(f'/content/tiny-imagenet-200/combined/{folder}')
		#for count in range(round(len(lis_file) *0.3)):
		files =random.sample(lis_file,round(len(lis_file) *0.3))
		for i in range(len(files)):
			file_path=f'/content/tiny-imagenet-200/combined/{folder}/' + files[i]
			dest=f'/content/tiny-imagenet-200/new_test/{folder}/'
			if os.path.isdir(dest)==False:
				os.mkdir(dest)
			shutil.move(file_path, dest)
#print(file_path)
    

