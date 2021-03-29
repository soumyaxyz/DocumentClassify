import cv2                        # working with, mainly resizing, images
import numpy as np                # dealing with arrays
import os                         # dealing with directories
from sklearn.utils import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm             # a nice pretty percentage bar for tasks.
from keras.utils import np_utils
# from keras.preprocessing.sequence import pad_sequences
import pdb 
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.tokenize import TweetTokenizer
import re
# from augmentations import image_transformation
# from models import get_model, model_change_softmax_size



def normalize(X, Y, num_classes):
	
	try:
		X = np.array(X) 
		Y = np_utils.to_categorical(Y,num_classes)     
		Y = np.array(Y)
			
		X, Y = shuffle(X, Y, random_state=0)

		X = X.astype('float32')
		X = X/255.0

	except Exception as e:
		print (e)
		pdb.set_trace()
		# pass
	
	return X, Y

def get_remapping(data_classes):
	LABEL_MAP = dict()
	for i, lbl, in enumerate(data_classes):
		LABEL_MAP[lbl] = i
	return LABEL_MAP

def tokenize(tokenizer, line):
	line = re.sub(r'\b\w{,3}\b', '', line) # removes words that are less than 4 characters 
	line = re.sub(r'\b\d+\b', 'NUM', line) # replaces number with NUM
	data = tokenizer.tokenize(line)
	# data = [word.lower() for word in data if word.isalnum()]
	# data = [word.lower() for word in data if word.isalpha()]
	data = [word if word == 'NUM' else word.lower() for word in data if word.isalpha()]



def create_CDIP_data(label_path, split_name, selected_classes = None, num_classes=16, DIR = 'RVL-CDIP'):
	print('Processing ' +split_name+'.txt')   
	label_file = os.path.join(label_path,split_name+'.txt')
	# split_name = split_name [:-4]
	try:
		if not selected_classes:
			X = np.load( os.path.join(DIR,split_name+'_saved_data_X.npy'),allow_pickle=True )
			Y = np.load( os.path.join(DIR,split_name+'_saved_data_Y.npy'),allow_pickle=True )
		else:
			X = np.load( os.path.join(DIR,split_name+'_6_saved_data_X.npy'),allow_pickle=True )
			Y = np.load( os.path.join(DIR,split_name+'_6_saved_data_Y.npy'),allow_pickle=True )
		return X,Y
	except Exception as e:
		print(e)

		IMG_DIR = os.path.join(DIR,'images/')
		if not selected_classes:
			num_classes = len(selected_classes)
			pdb.set_trace()
		X = []
		Y = []
		

		num_lines = sum(1 for line in open(label_file,'r'))
		with open(label_file) as f:
			for line in tqdm(f, total=num_lines):
			# line =   f.readline()
				img_path, lbl = line.strip().split()
				# lbl = int(lbl)
				if not selected_classes or lbl in selected_classes:
					try:
						img_path = os.path.join(IMG_DIR,img_path)

						img = cv2.imread(img_path,cv2.IMREAD_COLOR)
						img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH,))
						X.append(img)
						# LABEL_MAP[lbl]
						Y.append(lbl)
					except Exception as e:
						print(e)
						pass # just ignoring this sample
					

		
		     
		X, Y = normalize(X, Y, num_classes)

		if not selected_classes:
			np.save( os.path.join(DIR,split_name+'_saved_data_X.npy'), X)
			np.save( os.path.join(DIR,split_name+'_saved_data_Y.npy'), Y)
		else:
			X = np.load( os.path.join(DIR,split_name+'_6_saved_data_X.npy'),allow_pickle=True )
			Y = np.load( os.path.join(DIR,split_name+'_6_saved_data_Y.npy'),allow_pickle=True )

		return X,Y

def load_RVL_CDIP_data(selected_classes= None):
	DIR 		=  'RVL-CDIP'
	IMG_DIR 	= os.path.join(DIR,'images/')
	label_path 	= os.path.join(DIR,'labels/')

	
	
	val_X, val_Y 		= create_CDIP_data(label_path, 'val', selected_classes)	
	train_X, train_Y 	= create_CDIP_data(label_path, 'train', selected_classes)
	test_X, test_Y 		= create_CDIP_data(label_path, 'test', selected_classes)

	# pdb.set_trace()
	return train_X, train_Y, val_X, val_Y, test_X, test_Y

def create_NDLI_data(label_path, split_name, selected_classes, tokenizer, dicts, DIR = 'NDLI_data', IMG_HEIGHT = 224, IMG_WIDTH = 224, recompute = False, TEXT_LEN = 1000):
	print('Processing ' +split_name+'.txt')   
	label_file = os.path.join(label_path,split_name+'.txt')
	# split_name = split_name [:-4]
	n_class = '_11_class_'
	try:
		if recompute:
			raise ValueError('Recomputing ...')

		X = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_X.npy'),allow_pickle=True )
		T = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_T.npy'),allow_pickle=True )
		Y = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_Y.npy'),allow_pickle=True )		
		# dicts = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_dicts.npy'),allow_pickle=True )
		return X,T,Y
	except Exception as e:
		print(e)

		# IMG_DIR = os.path.join(DIR,'images/')
		IMG_DIR = DIR
		num_classes = len(selected_classes)
		X = []
		T = []
		Y = []
		[vocsize, idx2word, word2idx] = dicts 

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
						# pdb.set_trace()
						img = cv2.imread(img_path,cv2.IMREAD_COLOR)
						img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH,))


						with open(txt_path, 'r') as file:
							text_content = file.read().replace('\n', ' ')
						data = tokenize(tokenizer, text_content)

						indexed_data = []
						if data:								
							for word in data:
								try:
									index = word2idx[word]
								except:
									index = -1

								indexed_data.append(index)

						# padding to maxlen
						indexed_data += [0] * (TEXT_LEN - len(indexed_data))
						# truncating maxlen
						indexed_data = indexed_data[:TEXT_LEN]

						X.append(img)
						T.append(indexed_data)
						Y.append(lbl)
					except Exception as e:
						print(e)
						pass # just ignoring this sample
					

		
		     
		X = np.array(X) 
		T = np.array(T)
		Y = np_utils.to_categorical(Y,num_classes)     
		Y = np.array(Y)

		# pdb.set_trace()
			
		X, T, Y = shuffle(X, T, Y, random_state=0)

		X = X.astype('float32')
		X = X/255.0

		# pdb.set_trace()

		np.save( os.path.join(DIR,split_name+n_class+'_saved_data_X.npy'), X)
		np.save( os.path.join(DIR,split_name+n_class+'_saved_data_T.npy'), T)
		np.save( os.path.join(DIR,split_name+n_class+'_saved_data_Y.npy'), Y)
		# np.save( os.path.join(DIR,split_name+n_class+'_saved_data_dicts.npy'), dicts)


		return X,T,Y

def create_data_dict(label_path, selected_classes, tokenizer, DIR = 'NDLI_data'):
	label_file = os.path.join(label_path,'train.txt')
	words = set()
	with open(label_file) as f:
		for line in f : #tqdm(f, total=num_lines):
			try:
				img_path, lbl = line.strip().split()
			except Exception as e:
				pdb.set_trace()

			if lbl in selected_classes:
				try:
					img_path = os.path.join(DIR,img_path)
					txt_path = img_path.replace('images','texts')[:-4]+".txt"
					# pdb.set_trace()
					with open(txt_path, 'r') as file:
						text_content = file.read().replace('\n', ' ')
					data = tokenize(tokenizer, text_content)
					if data:
						for word in data:
							words.add(word) 
				except Exception as e:
					print(e)
					# pdb.set_trace()
					# return
					pass # just ignoring this sample

	words   = list(words)
	# for converting words to indices
	word2idx    = {w: i+1 for i, w in enumerate(words)}

	# for converting indices back to words
	idx2word    = dict((k,v) for v,k in word2idx.items())		
	idx2word[0]     = 'PAD'
	idx2word[-1]    = 'unknown'
	vocsize 		=  len(words)+1

	dicts = [vocsize, idx2word, word2idx]

	return dicts


def load_NDLI_data(NDLI_classes, recompute = False, TEXT_LEN = 200):
	DIR 		=  'NDLI_data'
	IMG_DIR 	= os.path.join(DIR,'images/')
	label_path 	= os.path.join(DIR,'labels/')

	tt = TweetTokenizer()	
	dicts = create_data_dict(label_path, NDLI_classes, tt, DIR)

	train_X, train_T, train_Y 	= create_NDLI_data(label_path, 'train', NDLI_classes, tt, dicts, DIR, recompute = recompute, TEXT_LEN = TEXT_LEN)
	val_X, val_T, val_Y 		= create_NDLI_data(label_path, 'val', 	NDLI_classes, tt, dicts, DIR, recompute = recompute, TEXT_LEN = TEXT_LEN)	
	test_X, test_T, test_Y 		= create_NDLI_data(label_path, 'test', 	NDLI_classes, tt, dicts, DIR, recompute = recompute, TEXT_LEN = TEXT_LEN)

	return train_X, train_T, train_Y, val_X, val_T, val_Y, test_X, test_T, test_Y, dicts




def load_NDLI_multi_page_data(NDLI_class_names, dicts, recompute = False, TEXT_LEN= 200):
	DIR 		=  'NDLI_data'
	# IMG_DIR 	= os.path.join(DIR,'images/multi-page/imgs')
	# label_path 	= os.path.join(DIR,'labels/')
	split_name 	= 'multi_page'
	IMG_HEIGHT = 224
	IMG_WIDTH = 224


	print('Processing ' +split_name)  

	n_class = '_11_class_'
	try:
		if recompute:
			raise ValueError('Recomputing ...')

		X = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_X.npy'),allow_pickle=True )
		T = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_T.npy'),allow_pickle=True )
		Y = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_Y.npy'),allow_pickle=True )		
		# dicts = np.load( os.path.join(DIR,split_name+n_class+'_saved_data_dicts.npy'),allow_pickle=True )
		return X,T,Y
	except Exception as e:
		print(e)

		IMG_DIR 	= os.path.join(DIR,'multi-page/imgs')
		[vocsize, idx2word, word2idx] = dicts

		tokenizer = TweetTokenizer()

		num_classes = len(NDLI_class_names)
		X = []
		T = []
		Y = []

		pic_dirs = os.listdir(IMG_DIR)
		pic_dirs.sort()
		for pic_dir in pic_dirs:
			lbl_found = False
			y = 0
			for lbl in NDLI_class_names:
				if lbl in pic_dir:
					Y.append(y) 
					lbl_found = True
					break 
				y +=1
			if not lbl_found:
				print('label not found for '+str(pic_dir))
				pdb.set_trace()
				# pass

			# pdb.set_trace() 
			x = [] 
			t = []
			pic_dir = os.path.join(IMG_DIR, pic_dir)
			pics = os.listdir(pic_dir)
			# pdb.set_trace()
			pics.sort()

			for i, pic in enumerate(pics):
				try:
					img_path = os.path.join(pic_dir, pic)
					txt_path = img_path.replace('imgs','texts')[:-4]+".txt"
					# pdb.set_trace()
					img = cv2.imread(img_path,cv2.IMREAD_COLOR)
					img = cv2.resize(img, (IMG_HEIGHT,IMG_WIDTH,))
					# print(img_path)



					with open(txt_path, 'r') as file:
						text_content = file.read().replace('\n', ' ')
					data = tokenize(tokenizer, text_content)

					indexed_data = []
					if data:								
						for word in data:
							try:
								index = word2idx[word]
							except:
								index = -1

							indexed_data.append(index)

					# padding to maxlen
					indexed_data += [0] * (TEXT_LEN - len(indexed_data))
					# truncating maxlen
					indexed_data = indexed_data[:TEXT_LEN]

					x.append(img)
					t.append(indexed_data)

				except Exception as e: 
					print(e)
					pass # just ignoring this sample

				#temp
				# if i>10:
				# 	break

			x = np.array(x)
			t = np.array(t)
			x = x.astype('float32')
			x = x/255.0
			X.append(x)	
			T.append(t)	

			print(str(pic_dir)+':'+str(i))
		
		     
		# X = np.array(X) 

		# pdb.set_trace()

		# Y = np_utils.to_categorical(Y,num_classes)     
		# Y = np.array(Y)

		# pdb.set_trace()
			
		# X, Y = shuffle(X, Y, random_state=0)

		# X = X.astype('float32')
		# X = X/255.0

		# pdb.set_trace()

		np.save( os.path.join(DIR,split_name+n_class+'_saved_data_X.npy'), X)		
		np.save( os.path.join(DIR,split_name+n_class+'_saved_data_T.npy'), T)
		np.save( os.path.join(DIR,split_name+n_class+'_saved_data_Y.npy'), Y)


	return X, T, Y














# def plot_triplets(examples):
#     plt.figure(figsize=(6, 2))
#     for i in range(3):
#         plt.subplot(1, 3, 1 + i)
#         plt.imshow(examples[i], cmap='binary')
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()




# plot_triplets([train_X[0], train_X[1], train_X[2]])



# def get_triple_idx(X, Y,  DIR = 'NDLI_data'):
# 	try:
# 		triplets_idx = np.load( os.path.join(DIR,'_saved_triplets_idx.npy'), allow_pickle=True )

# 	except Exception as e:
# 		y = np.argmax(Y, axis=1)
# 		nclass =  max(y)+1
# 		classwise_indices = []
# 		for cls in range (nclass):
# 			classwise_indices.append(np.where(y==cls)[0])

# 		triplets_idx = []

# 		print('Computing all possible triplets...')

# 		for cls in range (nclass):
# 			for anchor in classwise_indices[cls]:
# 				positives = classwise_indices[cls][ classwise_indices[cls]!=anchor]
# 				for positive in positives:
# 					for othr_cls in range (nclass):
# 						if othr_cls!= cls:
# 							for negative in classwise_indices[othr_cls]:
# 								triplets_idx.append([anchor, positive, negative ])
# 		print('Shuffling...')
# 		triplets_idx = shuffle(triplets_idx , random_state=0)

# 		np.save( os.path.join(DIR,'_saved_triplets_idx.npy'), triplets_idx)

# 	return triplets_idx


# def get_batch_of_triplets(X, triplets_idx, batch_idx, batch_size=64):
# 	start_idx = batch_idx * batch_size
# 	# batch = []
# 	anchor = []
# 	positive = []
# 	negative = []
# 	for i in range(batch_size):	
# 		j = start_idx+i	
# 		# triplet = [X[triplets_idx[j][0]], X[triplets_idx[j][1]], X[triplets_idx[j][2]]]
# 		anchor.append(triplets_idx[j][0])
# 		positive.append(triplets_idx[j][1])
# 		negative.append(triplets_idx[j][2])

# 	anchor = np.asarray(anchor)
# 	positive = np.asarray(positive)
# 	negative = np.asarray(negative)

# 	return [anchor, positive, negative]

# def data_generator(X, triplets_idx, batch_size=256, emb_size = 512):
# 	batch_idx = 0;
# 	while True:
# 		batch_idx += 1
# 		x = get_batch_of_triplets(X, triplets_idx, batch_idx, batch_size)
# 		y = np.zeros((batch_size, 3*emb_size))
# 		yield x, y