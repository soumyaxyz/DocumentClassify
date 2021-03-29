import cv2                        # working with, mainly resizing, images
import numpy as np                # dealing with arrays
import os                         # dealing with directories
from sklearn.utils import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm             # a nice pretty percentage bar for tasks.
from keras.utils import np_utils
import pdb 
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from augmentations import image_transformation
from models import * #get_model, model_change_softmax_size
from train_and_eval import * #train, evaluate, evaluate_ensemble, load_model
from auxilary_functions import * #
from data_loader import *
import tensorflow as tf
import keras
from keras_self_attention import SeqSelfAttention





def run_baseline_experiment(modelname, init_weights='imagenet', embed_top_n= 100000, n_epochs = 20):

	save_location = os.path.join('Simple_runs', modelname)
	if not os.path.exists(save_location):
		os.makedirs(save_location)

	model = get_model(modelname, vocsize, idx2word, IMG_HEIGHT, IMG_WIDTH, TEXT_LEN, softmax_size = num_classes, trainable=False, weights=init_weights, embed_top_n = embed_top_n)
	# model_imagenet = get_model_as_is(modelname, IMG_HEIGHT,IMG_WIDTH, softmax_size = num_classes, trainable=True, weights=None)
	model_NDLI, hist, time_taken = train(model, train_X, train_T, train_Y, val_X, val_T, val_Y, save_location,  data_name = 'NDLI', n_epochs=n_epochs)
	print('Without pretraning on RVL-CDIP\n')
	loss, acc = evaluate(model_NDLI, test_X, test_T, test_Y)
	write_test_results(save_location+', NDLI, '+str(loss)+', '+str(acc))
	return model_NDLI, acc



def majority_vote(Y, num_classes):
	votes = [0]*num_classes
	Y =  np.round(Y)
	Y = Y.astype('int')
	for y in Y:
		try:
			votes[np.argmax(y)] +=1
		except Exception:
			pdb.set_trace()


	
	pred_at_0 = np.argmax(Y[0])

	pred = np.argmax(votes)
	conf =  max(votes)/sum(votes)

	pred_2 = np.argmax( votes[:pred]+[0]+votes[pred+1:] )
	conf_2 = votes[pred_2]/sum(votes)
	if pred == pred_2: # error condition
		pdb.set_trace()
	# print(votes)
	return pred, conf, pred_2, conf_2, pred_at_0, votes


def apply_expertize(pred, conf, pred_2, conf_2, pred_at_0):
	if pred == 10:  # Scientific
		if pred_at_0 == 5:  # Scientific with first page  Thesis
		 	pred = 5        # is set to Thesis
	elif pred == 0: # Catalog
		if pred_2 == 2 and conf - conf_2 < 0.1 :  # Catalog with aproximately equal probablity with Map
			pred = 2        # is set to Map


	return pred


TRAIN_DIR = './NDLI_data/Train/'  #directory containing training dataset
TEST_DIR = './NDLI_data/Test/'   #directory containing testing dataset
CDIP_DIR = './CDIP_data/'

IMG_HEIGHT = 224  #image height
IMG_WIDTH = 224 #image width
TEXT_LEN = 200
# LR = 1e-4      #learning rate
TRAIN_AFRESH = True
CONTINUE_TRAINING = False
NDLI_class_names = ["Catalog","Handwritten","Maps","Paintings","Presentation","Thesis","Law_Report","Music_Notations", "Newspaper_Article", "Question_Papers",  "Scientific"]
NDLI_classes = ['0','1','2', '3', '4', '5', '6', '7', '8', '9', '10']

num_classes = len(NDLI_classes)






(train_X, train_T, train_Y, val_X, val_T, val_Y, test_X, test_T, test_Y, dicts) = load_NDLI_data(NDLI_classes, recompute = False , TEXT_LEN = TEXT_LEN)
[vocsize, idx2word, word2idx] = dicts 
# pdb.set_trace()

(X, T, Y) = load_NDLI_multi_page_data(NDLI_class_names, dicts, recompute = False, TEXT_LEN = TEXT_LEN)
# pdb.set_trace()


# pdb.set_trace()

# modelnames = ['MobileNetV2', 'MobileNet', 'Xception','VGG16', 'VGG19']  #['MobileNetV2', 'MobileNet', 'Xception', 'VGG16', 'VGG19']
modelnames = ['VGG16']		


model = None 
model = keras.models.load_model('BEST_MODEL', custom_objects={'SeqSelfAttention': SeqSelfAttention})

if model:
	CM = generate_CM(model, test_X, test_T, test_Y, NDLI_class_names)
else:
	write_test_results('Model name, test dataset, Loss, Accuracy')
	for modelname in modelnames:
		print('\n\n\nTesting with :'+modelname)
		model, text_acc = run_baseline_experiment(modelname, n_epochs = 20)#, "imagenet")

		# pdb.set_trace()
		# pred_Y = model.predict_classes(test_X)  # sequential model
		pred_Y =  model.predict([test_X, test_T])



		CM = generate_CM(model, test_X, test_T, test_Y, NDLI_class_names)


		file1 = open("Results.txt", "a")  # append mode 
		file1.write(modelname +':\t'+str(text_acc)+'\n') 
		file1.close()


pdb.set_trace()



# run_siamize_training(embedding_model)
score = 0
corrected_score = 0	
Y_p = []
Y_a = []
page_count  = 0

for i, y in enumerate(Y):
	y_p = model.predict([X[i], T[i] ])
	pred, conf, pred_2, conf_2, pred_at_0, votes = majority_vote(y_p, num_classes)
	prediction = apply_expertize(pred, conf, pred_2, conf_2, pred_at_0)
	Y_p.append(pred)
	Y_a.append(prediction)

	pages = X[i].shape[0]+1
	page_count += pages
	print(str(NDLI_class_names[y])+"\tas "+str(NDLI_class_names[prediction])+" ("+str( pages )+" pages)"+
			"\n\t\tas "+str(NDLI_class_names[pred])+" with confidence: "+str(conf)[:4]+ 
			"\n\t\tas "+str(NDLI_class_names[pred_2])+" with confidence: "+str(conf_2)[:4]+ 
			"\n \t\tpred_at_0 :" + str(NDLI_class_names[pred_at_0]) ) # Votes:"+str(votes)+", 

	if pred == y:
		score += 1
	if prediction == y:
		corrected_score += 1
score =  score/ len(X)
corrected_score = corrected_score/ len(X)



# yT = np.argmax(test_Y, axis=1)
# yP = np.argmax(pred_Y, axis=1)
try:
	class_names = [NDLI_class_names[index] for index in set(Y)]	 
	CM_p = confusion_matrix(Y,Y_p)
	plot_CM(CM_p, class_names, 'CM_mp_p', size=30)

	CM_a = confusion_matrix(Y,Y_a)
	plot_CM(CM_a, class_names, 'CM_mp_a', size=30)
except Exception as e:
	print(e )

print(score, corrected_score , page_count ) 

pdb.set_trace()
import code
code.interact(banner="Start", local=locals())


# model.save("BEST_MODEL")


	
# # pdb.set_trace()


