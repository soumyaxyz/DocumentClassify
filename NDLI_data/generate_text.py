import cv2                        # working with, mainly resizing, images
import numpy as np                # dealing with arrays
import os                         # dealing with directories
from sklearn.utils import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm   
import pytesseract
import pdb
from keras.utils import np_utils					


def create_NDLI_data(label_path, split_name, selected_classes, DIR = '.', IMG_HEIGHT = 224, IMG_WIDTH = 224, recompute = False):
	print('Processing ' +split_name+'.txt')   
	label_file = os.path.join(label_path,split_name+'.txt')
	# split_name = split_name [:-4]
	n_class = '_11_class_'
	try:
		if recompute:
			raise ValueError('Recomputing ...')

		X = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_X.npy'),allow_pickle=True )
		Y = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_Y.npy'),allow_pickle=True )
		return X,Y
	except Exception as e:
		print(e)

		# IMG_DIR = os.path.join(DIR,'images/')
		IMG_DIR = DIR
		num_classes = len(selected_classes)
		X = []
		Y = []
		

		num_lines = sum(1 for line in open(label_file,'r'))
		with open(label_file) as f:
			for line in tqdm(f, total=num_lines):
			# line =   f.readline()
				try:
					img_path, lbl = line.strip().split()
				except Exception as e:
					pdb.set_trace()
				# pdb.set_trace()
				
				# lbl = int(lbl)
				if lbl in selected_classes:
					try:
						img_path = os.path.join(IMG_DIR,img_path)
						txt_path = img_path.replace('images','texts')[:-4]+".txt"
						# pdb.set_trac e()
						img = cv2.imread(img_path,cv2.IMREAD_COLOR)
						text = pytesseract.image_to_string(img)
						f = open(txt_path, "w")
						f.write(text)
						f.close()
						# img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH,))
						# X.append(img)
						# Y.append(lbl)
					except Exception as e:
						print(e)
						pass # just ignoring this sample
					

		
		     
		# X = np.array(X) 
		# Y = np_utils.to_categorical(Y,num_classes)     
		# Y = np.array(Y)

		# # pdb.set_trace()
			
		# X, Y = shuffle(X, Y, random_state=0)

		# X = X.astype('float32')
		# X = X/255.0

		# pdb.set_trace()

		# np.save( os.path.join(DIR,split_name+n_class+'_saved_data_X.npy'), X)
		# np.save( os.path.join(DIR,split_name+n_class+'_saved_data_Y.npy'), Y)


		return X,Y

def load_NDLI_data(NDLI_classes, recompute = False):
	DIR 		=  '.'
	IMG_DIR 	= os.path.join(DIR,'images/')
	label_path 	= os.path.join(DIR,'labels/')

	
	
	val_X, val_Y 		= create_NDLI_data(label_path, 'val', 	NDLI_classes, DIR, recompute = recompute)	
	train_X, train_Y 	= create_NDLI_data(label_path, 'train', NDLI_classes, DIR, recompute = recompute)
	test_X, test_Y 		= create_NDLI_data(label_path, 'test', 	NDLI_classes, DIR, recompute = recompute)

	return train_X, train_Y, val_X, val_Y, test_X, test_Y




NDLI_classes = ['0','1','2', '3', '4', '5', '6', '7', '8', '9', '10']
load_NDLI_data(NDLI_classes, recompute = False)
pdb.set_trace()