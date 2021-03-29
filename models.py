import keras as K
K.backend.clear_session()

import tensorflow as tf
import keras.applications as kerasapp # import VGG16, VGG16
from keras import optimizers,utils
from keras.models import Sequential, Model, load_model
from keras.models import  Sequential
from keras.layers import Input    
from keras.layers import Lambda, Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D,  LSTM, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import   Conv2D, MaxPooling2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import np_utils
from keras.utils import plot_model
from keras_self_attention import SeqSelfAttention
import embeddings_loader
import matplotlib.pyplot as plt
import traceback, pdb


	
# Check the trainable status of the individual layers
# for layer in vgg_model.layers:
#     print(layer, layer.trainable)


#model.summary()
def get_pre_trained(modelname, input_shape , trainable, weights='imagenet'):
	if modelname == 'Xception':
		pre_trained_model = kerasapp.Xception(include_top=False, weights=weights,  input_shape=input_shape )
	elif modelname == 'MobileNetV2':
		pre_trained_model = kerasapp.MobileNetV2(include_top=False, weights=weights,  input_shape=input_shape )
	elif modelname == 'MobileNet':
		pre_trained_model = kerasapp.MobileNet(include_top=False, weights=weights,  input_shape=input_shape )
	elif modelname == 'VGG19':
		pre_trained_model = kerasapp.VGG19(include_top=False, weights=weights,  input_shape=input_shape )
	else: # default : VGG16
		pre_trained_model = kerasapp.VGG16(include_top=False, weights=weights,  input_shape=input_shape )

	for layer in pre_trained_model.layers[:]:
		layer.trainable = trainable
	print("\n\nloading"+modelname+"\n\n")
	return pre_trained_model


def get_model_old(modelname, img_height, img_width, softmax_size = 10, trainable = False, weights='imagenet'):
	# Create the model
	model = Sequential()
	 

	input_shape = (img_height, img_width, 3)
	# Add the convolutional base model
	model.add(get_pre_trained(modelname, input_shape, trainable,  weights))		 #model.add(vgg_model)
	 
	# Add new layers
	model.add(Flatten())  #input_shape=train_data.shape[1:]))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(softmax_size, activation='softmax', name='output_layer'))
	 
	# Show a summary of the model. Check the number of trainable parameters
	model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)
	print('trainable =', trainable)

	#model.compile(loss='categorical_crossentropy',
	#            optimizer='rmsprop',
	#              metrics=['accuracy'])

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	

	return model



def get_model(modelname, vocsize, idx2word, img_height =224, img_width=224, text_len=1000,  lstm_dim = 128, softmax_size = 10, trainable = False, weights='imagenet', embed_top_n = -1):
	embeddings = embeddings_loader.glove_embeddings_loader(embed_top_n)

	# pdb.set_trace()

	# Create the model	
	 
	input_shape_A = (img_height, img_width, 3)
	input_A = Input(shape=input_shape_A)

	input_shape_B = (text_len,)
	input_B = Input(shape=input_shape_B)

	# Add the convolutional base model
	pre_trained_model = get_pre_trained(modelname, input_shape_A, trainable,  weights)
	pre_trained_model = pre_trained_model(input_A)

	# Add new layers
	L1 = Flatten()(pre_trained_model)
	L2 = Dense(1024, activation='relu')(L1)
	L2 = Dropout(0.1)(L2)
	L3 = Dense(512, activation='relu')(L2)
	L3 = Dropout(0.1)(L3)

	embeds, _, _  = embeddings.init_weights(idx2word)		
	embed = Embedding(input_dim= vocsize, output_dim= embeddings.embed_dim, input_length= text_len, weights=[embeds], mask_zero=False, name='embedding', trainable=True)(input_B)
	blstm_layer = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))(embed)
	attention_layer =SeqSelfAttention(attention_activation='sigmoid')(blstm_layer)

	attention_layer = Flatten()(attention_layer)
	LA = Dense(128, activation='relu')(attention_layer)

	L4 = K.layers.Concatenate()([L3, LA])

	L4 = Dense(512, activation='relu')(L4)
	L4 = Dropout(0.1)(L4)
	out = Dense(softmax_size, activation='softmax', name='output_layer')(L4)

	model = Model(inputs=[input_A, input_B],outputs=out)

	
	
	 
	# Show a summary of the model. Check the number of trainable parameters
	model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)

	print('trainable =', trainable)

	#model.compile(loss='categorical_crossentropy',
	#            optimizer='rmsprop',
	#              metrics=['accuracy'])

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	

	return model


def get_model_as_is(modelname, img_height, img_width, softmax_size = 10, trainable = True, weights='imagenet'):
	model =  get_pre_trained(modelname, img_height, img_width, trainable,  weights)
	model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)
	print('trainable =', trainable)

	#model.compile(loss='categorical_crossentropy',
	#            optimizer='rmsprop',
	#              metrics=['accuracy'])

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


def model_change_softmax_size(model, new_size):
	# pdb.set_trace()
	model.pop()
	model.add(Dense(new_size, activation='softmax', name='output_layer'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model




# for layer in model.layers:
#     print(layer)

# def build_alt_network(input_shape, embeddingsize):
# 	'''
# 	Define the neural network to learn image similarity
# 	Input : 
# 			input_shape : shape of input images
# 			embeddingsize : vectorsize used to encode our picture   
# 	'''
# 	 # Convolutional Neural Network
# 	network = Sequential()
# 	network.add(Conv2D(128, (7,7), activation='relu',
# 					 input_shape=input_shape,
# 					 kernel_initializer='he_uniform',
# 					 kernel_regularizer=l2(2e-4)))
# 	network.add(MaxPooling2D())
# 	network.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
# 					 kernel_regularizer=l2(2e-4)))
# 	network.add(MaxPooling2D())
# 	network.add(Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',
# 					 kernel_regularizer=l2(2e-4)))
# 	network.add(Flatten())
# 	network.add(Dense(4096, activation='relu',
# 				   kernel_regularizer=l2(1e-3),
# 				   kernel_initializer='he_uniform'))
	
	
# 	network.add(Dense(embeddingsize, activation=None,
# 				   kernel_regularizer=l2(1e-3),
# 				   kernel_initializer='he_uniform'))
	
# 	#Force the encoding to live on the d-dimentional hypershpere
# 	network.add(Lambda(lambda x: K.backend.l2_normalize(x,axis=-1)))
	
# 	return network

# def triplet_loss(y_true, y_pred):
# 	emb_size = 64
# 	alpha = 0.2
# 	anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
# 	positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
# 	negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
# 	return tf.maximum(positive_dist - negative_dist + alpha, 0.)


# def get_siamize_net(embedding_model, input_shape):

# 	# for embedding_model the final sodtmax layer must be removed
# 	embedding_model.pop()

# 	input_anchor = tf.keras.layers.Input(shape=input_shape)
# 	input_positive = tf.keras.layers.Input(shape=input_shape)
# 	input_negative = tf.keras.layers.Input(shape=input_shape)

# 	embedding_anchor = embedding_model(input_anchor)
# 	embedding_positive = embedding_model(input_positive)
# 	embedding_negative = embedding_model(input_negative)

# 	output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

# 	siamize_net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
# 	siamize_net.summary()

# 	siamize_net.compile(loss=triplet_loss, optimizer='adam')

# 	return siamize_net


# old implementation

# class TripletLossLayer(Layer):
# 	def __init__(self, alpha, **kwargs):
# 		self.alpha = alpha
# 		super(TripletLossLayer, self).__init__(**kwargs)
	
# 	def triplet_loss(self, inputs):
# 		try:
# 			anchor, positive, negative = inputs
# 			p_dist = K.backend.sum(K.backend.square(anchor-positive), axis=-1)
# 			n_dist = K.backend.sum(K.backend.square(anchor-negative), axis=-1)
# 			return K.backend.sum(K.backend.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
# 		except Exception as e:
# 			pdb.set_trace()
		

	
# 	def call(self, inputs):
# 		loss = self.triplet_loss(inputs)
# 		self.add_loss(loss)
# 		return loss

# def get_siamize(input_shape, network, margin=0.2):
# 	'''
# 	Define the Keras Model for training 
# 		Input : 
# 			input_shape : shape of input images
# 			network : Neural network to train outputing embeddings
# 			margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
	
# 	'''
# 	 # Define the tensors for the three input images

# 	try:
		
		
# 		anchor_input = Input(input_shape, name="anchor_input")
# 		positive_input = Input(input_shape, name="positive_input")
# 		negative_input = Input(input_shape, name="negative_input") 
		
# 		# Generate the encodings (feature vectors) for the three images
# 		encoded_a = network(anchor_input)
# 		encoded_p = network(positive_input)
# 		encoded_n = network(negative_input)
		
# 		#TripletLoss Layer
# 		loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
		
# 		# Connect the inputs with the outputs
# 		network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
		
# 		# return the model
# 		return network_train
# 	except Exception as e:
# 		traceback.print_exc()
# 		pdb.set_trace()