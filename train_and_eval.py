import time
import datetime  
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import traceback, pdb
from auxilary_functions import * 






def train(model, train_X, train_T, train_Y, val_X, val_T, val_Y, modelname,  data_name = '', n_epochs=50):
	t=time.time()
	callbacks = [	EarlyStopping(monitor='val_loss', patience=2), 
					ModelCheckpoint(filepath=modelname+'/'+data_name+'_best_model.ckpt', monitor='val_loss', save_best_only=True)
				]

	hist = model.fit([train_X, train_T] , train_Y, batch_size=64, epochs=n_epochs, verbose=1, validation_data=([val_X, val_T], val_Y),callbacks=callbacks)


	# pdb.set_trace()

	try:
		plot_traning_summary(hist)


		t1=time.time()
		time_taken = convert_timestamp_difference_to_human_readable(t1-t)

		print('Training time: %s Hrs' % time_taken)

		print('\n\n')


		save_run_hist(modelname, hist, time_taken)

	except Exception as e:
		traceback.print_exc()
		pdb.set_trace()
	


	return model, hist, time_taken

def load_model(modelname, data_name = '') :
	from keras.models import load_model
	return load_model(modelname+'/'+data_name+'_best_model.ckpt')

def save_run_hist(modelname, hist, time_taken):
	hist_df = pd.DataFrame(hist.history)
	time_df = pd.DataFrame({'total time':[time_taken]})
	combined = pd.concat([hist_df, time_df], axis=1)
	hist_csv_file = modelname+'/history.csv'
	with open(hist_csv_file, mode='a') as f:
		combined.to_csv(f)



# def save_model(model, modelname, dataset) :  
# 	model.save(modelname+'/'+dataset+'_model.hp5')


def evaluate(model, test_X, test_T, test_Y):
	(loss, accuracy) = model.evaluate([test_X, test_T],  test_Y, batch_size=64, verbose=1)
	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
	return loss,accuracy


def evaluate_ensemble(model, test_X, test_Y, num_aug_classes, num_classes, show_details=False):


	yPred = []
	for aug_class in range(num_aug_classes):
		yPred.append( model.predict(test_X[aug_class], verbose=1) )

	
	pred = []
	pred_0 = []
	gold = []
	num_test_sample = len(test_Y[0])
	for test_idx in range(num_test_sample):
		pred_votes = [0]*num_classes
		all_pred = []
		for aug_class in range(num_aug_classes):
			p_aug_class =  np.argmax(yPred[aug_class][test_idx])
			pred_votes[p_aug_class] += 1
			all_pred.append(p_aug_class)

		ensbl_pred = np.argmax(pred_votes)
		flag = 'E'
		if pred_votes[ensbl_pred]/num_aug_classes <.66:   # only if supermajority disagrees, is the basic_pred over written
			ensbl_pred = np.argmax(yPred[0][test_idx])
			flag = ' '

		if show_details:
			if np.argmax(pred_votes) != np.argmax(test_Y[0][test_idx]):  # wrong
				if np.argmax(pred_votes) != np.argmax(yPred[0][test_idx]):  #ensemble dis-agrement
					print(np.argmax(test_Y[0][test_idx]), np.argmax(yPred[0][test_idx]), ensbl_pred, flag, pred_votes  )
				else:   #ensemble agrement
					print(np.argmax(test_Y[0][test_idx]), np.argmax(yPred[0][test_idx]), ensbl_pred, ' ', pred_votes  )
			else: # correct
				if np.argmax(pred_votes) != np.argmax(yPred[0][test_idx]):  #ensemble dis-agrement
					print(np.argmax(test_Y[0][test_idx]), np.argmax(yPred[0][test_idx]), '-', flag, pred_votes  )
				else:   #ensemble agrement
					print( np.argmax(test_Y[0][test_idx]) )


		pred.append(ensbl_pred)
		pred_0.append(np.argmax(yPred[0][test_idx]))
		gold.append(np.argmax(test_Y[0][test_idx]))

	ensemble_accuracy  = accuracy_score(gold, pred)
	 
	print("[INFO] accuracy: {:.4f}%".format(accuracy_score(gold, pred_0) * 100))
	print("[INFO] ensemble accuracy: {:.4f}%".format(ensemble_accuracy* 100))
	return ensemble_accuracy


	# # test_idx = 26

	# # Showing the incorrect predictions
	# print('test_idx: \t Actual \t\t: ','Predicted:')
	# for test_idx in range(num_test_sample):

	#   if pred[test_idx] != gold[test_idx]:
	#     print(test_idx,':\t ', selected_classes[gold[test_idx]][:12],'\t\t: ',selected_classes[pred[test_idx]][:12])
	#   else:
	#     print(test_idx,':\t ', selected_classes[gold[test_idx]][:12])
	#     # plt.imshow(X_test2[test_idx])

	# test_idx = 5
	# print('Actual \t\t: ',selected_classes[gold[test_idx]])
	# print('Predicted \t: ',selected_classes[pred[test_idx]])
	# plt.imshow(test_X[0][test_idx])

	# plt.show()