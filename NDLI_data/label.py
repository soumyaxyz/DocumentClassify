import os
from shutil import copyfile
import random


base_dir 	= 'images'
out_dir 	= 'labels'
class_dirs 	= os.listdir(base_dir) 

if not os.path.exists(out_dir):
	os.makedirs(out_dir)


training_file = open(os.path.join(out_dir, 'train.txt'), "w")  
testing_file = open(os.path.join(out_dir, 'test.txt'), "w")  

counts = dict()
for label, class_dir in enumerate (class_dirs):
	class_dir_full = os.path.join(base_dir, class_dir)
	if os.path.isdir(class_dir_full):

		count = 0

		path_and_labels = []

		files = os.listdir(class_dir_full)
		# count = len( files[0].split('_') )
		for file in files:
			file_path = os.path.join(class_dir_full, file)
			if not os.path.isdir(file_path):
				
				count += 1

				path_and_labels.append(file_path+' '+str(label))

				# print(file_path,label)

				# if file.endswith('1.jpeg'):
				# # if len( file.split('_') ) == count:
				# 	copyfile(os.path.join(class_dir_full, file), os.path.join(out_class_dir_full, file))

		counts[class_dir] 	=  count
		# count_train 		=  round(count*.55)
		# count_test 		= round(count*.45)
		count_train 		= 100
		count_test 			= count - count_train


		print(class_dir, count, count_train, count_test)

		assert round(count*.55) + round(count*.45) == count

		

		# randomizing
		path_and_labels 	= random.sample(path_and_labels,  count  )

		for path_and_label in path_and_labels[:count_train]:
			training_file.write(path_and_label+'\n')
		# print('\n')
		for path_and_label in path_and_labels[count_train:]:
			testing_file.write(path_and_label+'\n')

		
		


		# print('\n\n')
# print(counts)

training_file.close()
testing_file.close()