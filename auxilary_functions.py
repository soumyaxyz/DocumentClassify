import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd
import traceback, pdb
import time
import datetime  
import numpy as np 
import copy

def plot_image(X, Y):
	# X = train_X[i]
	# Y = train_Y[i]
	fig = plt.figure(figsize=(3, 6))
	plt.imshow(X)
	fig.savefig('a_fig.png')
	print(NDLI_class_names[train_Y[0]])
	print(NDLI_class_names[np.argmax(Y)])


def plot_traning_summary(hist):
	#lets plot the train and val curve
	#get the details form the history object
	try:
		acc = hist.history['accuracy']
		val_acc = hist.history['val_accuracy']
	except Exception as e:
		acc = hist.history['acc']
		val_acc = hist.history['val_acc']
	
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']

	epochs = range(1, len(acc) + 1)

	#Train and validation accuracy
	plt.plot(epochs, acc, 'b', label='Training accurarcy')
	plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
	plt.title('Training and Validation accurarcy')
	plt.legend()
	plt.savefig('Training_and_Validation_acc.png')
	plt.figure()
	#Train and validation loss
	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and Validation loss')
	plt.legend()
	plt.savefig('Training_and_Validation_loss.png')
	plt.show()


def convert_timestamp_difference_to_human_readable(n):
	n = round(n)
	return str(datetime.timedelta(seconds = n))


def print_shapes(CDIP = True):
	if CDIP:
		print('\n\nCDIP shapes:')
		print(pre_train_X.shape)
		print(pre_train_Y.shape)
		print(pre_val_X.shape)
		print(pre_val_Y.shape)
		print(pre_test_X.shape)
		print(pre_test_Y.shape)
	print('\n\nNDLI shapes:')
	print(train_X.shape)
	print(train_Y.shape)
	print(val_X.shape)
	print(val_Y.shape)
	print(test_X.shape)
	print(test_Y.shape)

def write_test_results(message):
	f = open("test_results.csv", "a")
	f.write(message+'\n')
	f.close()

def plot_CM(CM, NDLI_classes, save_file_name = 'CM', cbar_flag = False, size=14 ):		
	NDLI_labels = []
	for i in range(len(NDLI_classes)):
		# NDLI_REMAP[str(i)]
		label = NDLI_classes[i]
		label = label.replace('_',' ')
		NDLI_labels.append(label)
	CM_orig = copy.deepcopy(CM)
	if not cbar_flag:
		for i in range(len(CM)):
			CM[i] =  np.round(CM[i]/sum(CM[i])*100,2)


	df_cm = pd.DataFrame(CM, NDLI_labels, NDLI_labels)

	
	# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
	df_cm.index.name = 'Actual'
	df_cm.columns.name = 'Predicted' 
	plt.figure(figsize = (15,15))
	sn.set(font_scale=size/12)#for label size

	chart = sn.heatmap(df_cm, annot=CM_orig, cmap="Blues", fmt='g', cbar=cbar_flag, annot_kws={"size": size})# font size

	if cbar_flag:
		max_val =  np.sum(CM[-1])
		cbar = chart.collections[0].colorbar
		cbar.set_ticks([0, max_val*.25, max_val*.50, max_val*.75, max_val])
		cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
	else:
		pass # for t in chart.texts: t.set_text(t.get_text() + "%")

	plt.yticks(rotation= 0) #, ha="left")
	plt.xticks(rotation= 90)
	plt.savefig(save_file_name+'.png') 
	# print("fuck you !!")
	# pdb.set_trace()

def print_class_wise_recall(CM, NDLI_classes):
	for i in range(len(CM)):
		try:
			# NDLI_REMAP[str(i)]
			if sum(CM[i])>0:
				print(NDLI_classes[i],'\t: ',round(CM[i,i]*100/sum(CM[i]),2),'%')
		except Exception as e:
			pass

def generate_CM(model, test_X, test_T, test_Y, NDLI_class_names):
	pred_Y = model.predict([test_X, test_T], verbose=1)
	yT = np.argmax(test_Y, axis=1)
	yP = np.argmax(pred_Y, axis=1)
	
	CM = confusion_matrix(yT,yP)

	print_class_wise_recall(CM, NDLI_class_names)
	plot_CM(CM, NDLI_class_names)

	return  CM

def class_merge_exp(model, test_X, test_Y):
	pred_Y = model.predict(test_X, verbose=1)
	yT = np.argmax(test_Y, axis=1)
	yP = np.argmax(pred_Y, axis=1)

	yT[yT==2] = 4
	yT[yT==9] = 10
	yP[yP==2] = 4
	yP[yP==9] = 10

	print('[INFO] accuracy: ',accuracy_score(yT,yP)*100,'%')
	
	CM = confusion_matrix(yT,yP)
	
	NDLI_merged_classes = ['Advertisements', 'Catalog', 'Handwritten', 'Dissertations+\nLaw_report', 'Leaflet', 'Map', 'Music_notation', 'News_article', 'Painting+\nPhotograph', 'Presentation', 'Questionaire', 'Scientific']

	print_class_wise_recall(CM, NDLI_merged_classes)
	plot_CM(CM, NDLI_merged_classes,  save_file_name = 'merged_class_CM', cbar_flag = False)

	return  CM
