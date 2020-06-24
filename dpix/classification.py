import representation
import numpy
import random
import matplotlib.pyplot as plt

def extract_data_for_cnn(real_data, train_proportion):
	##
	## Prepare data to fed the CNN, real_data is a data structure generated
	## by build_patient_matrix function from the representation module
	## train_proportsion is a float between 0 and 1, represent the proportion of
	## the training set among all data
	##
	## Perform a few operation to preprocess the data
	##
	## return a tuple of 2 tuples : (tran_X, train_Y) and (test_X, test_Y)
	##

	## Get data and labels
	all_data_vector_X = []
	all_data_vector_Y = []
	for observation in real_data.keys():
		value = real_data[observation]
		all_data_vector_X.append(value[0])
		all_data_vector_Y.append(value[1])

	## split to train and test set
	number_of_observation = len(all_data_vector_Y)
	train_size = int(train_proportion * number_of_observation)
	number_of_patients_in_training_set = 0
	patients_index_in_training_set = []

	train_data_vector_X = []
	train_data_vector_Y = []
	test_data_vector_X = []
	test_data_vector_Y = []

	while(number_of_patients_in_training_set != train_size):
		candidate = random.randint(0, len(all_data_vector_X)-1)
		if(candidate not in patients_index_in_training_set):
			train_data_vector_X.append(all_data_vector_X[candidate])
			train_data_vector_Y.append(all_data_vector_Y[candidate])
			number_of_patients_in_training_set += 1
			patients_index_in_training_set.append(candidate)

	for x in range(0, len(all_data_vector_X)):
		if(x not in patients_index_in_training_set):
			test_data_vector_X.append(all_data_vector_X[x])
			test_data_vector_Y.append(all_data_vector_Y[x])

	## cast vector into numpy array
	train_data_vector_X = numpy.array(train_data_vector_X)
	train_data_vector_Y = numpy.array(train_data_vector_Y)
	test_data_vector_X = numpy.array(test_data_vector_X)
	test_data_vector_Y = numpy.array(test_data_vector_Y)

	## convert each matrix of the train and test set
	## into a matrix of size heigh x width x 1 to be fed into
	## the network
	side_size_x = len(train_data_vector_X[0][0])
	side_size_y = len(train_data_vector_X[0])
	train_data_vector_X = train_data_vector_X.reshape(-1, side_size_x,side_size_y, 1)
	test_data_vector_X = test_data_vector_X.reshape(-1, side_size_x,side_size_y, 1)

	## convert int8 format type to float32
	## rescale the pixel values in range 0 - 1 inclusive
	train_data_vector_X = train_data_vector_X.astype('float32')
	test_data_vector_X = test_data_vector_X.astype('float32')
	train_data_vector_X = train_data_vector_X / 255.
	test_data_vector_X = test_data_vector_X / 255.

	## return train and test vector with associated labels vector
	return ((train_data_vector_X, train_data_vector_Y), (test_data_vector_X, test_data_vector_Y))



def prepare_prediction_dataset_for_cnn(real_data):
	##
	## Prepare prediction data for the CNN
	## based on the extract_data_for_cnn function
	## but do not care about labels (Y vectors)
	##
	## real_data is a data_structure genrated by the
	## build_prediction_matrix function from the
	## representation package
	##

	## Get data and labels
	all_data_vector_X = []
	for observation in real_data.keys():
		value = real_data[observation]
		all_data_vector_X.append(value[0])

	## cast vector into numpy array
	prediction_data_vector_X = numpy.array(all_data_vector_X)

	## convert each matrix of the train and test set
	## into a matrix of size heigh x width x 1 to be fed into
	## the network
	side_size_x = len(prediction_data_vector_X[0][0])
	side_size_y = len(prediction_data_vector_X[0])
	prediction_data_vector_X = prediction_data_vector_X.reshape(-1, side_size_x,side_size_y, 1)

	## convert int8 format type to float32
	## rescale the pixel values in range 0 - 1 inclusive
	prediction_data_vector_X = prediction_data_vector_X.astype('float32')
	prediction_data_vector_X = prediction_data_vector_X / 255.

	## return train and test vector with associated labels vector
	return prediction_data_vector_X





def run_CNN(train_X, train_Y, test_X, test_Y, epochs, prediction_set, grid_matrix):
	##
	## IN PROGRESS
	##

	## importation
	import keras
	from keras.models import Sequential,Input,Model
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers.normalization import BatchNormalization
	from keras.layers.advanced_activations import LeakyReLU
	from keras.utils import to_categorical
	from sklearn.model_selection import train_test_split

	import h5py

	from vis.visualization import visualize_saliency
	from vis.utils import utils
	from keras import activations
	from vis.visualization import visualize_cam

	import numpy as np
	from sklearn import metrics

	import gradient_visualisation

	## Change the labels from categorical to one-hot encoding
	train_Y_one_hot = to_categorical(train_Y)
	test_Y_one_hot = to_categorical(test_Y)

	## split in train and validation set
	## keep the test set for the end
	train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

	## Model param
	batch_size = 64
	input_size_x = len(train_X[0][0])
	input_size_y = len(train_X[0])

	## get number of classes
	class_list = []
	observation_file = open("observations_classification.csv", "r")
	for line in observation_file:
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		class_id = line_in_array[2]
		if(class_id not in class_list):
			class_list.append(class_id)
	observation_file.close()
	num_classes = len(class_list)

	##---------------------------##
	## Build and train the model ##
	##---------------------------##


	## DEBUG
	optimisation = False

	if(optimisation):

		print("[HYPERPARAMETRISATION][IN PROGRESS] => Craft a CNN Architecture")

		## IN PROGRESS
		## Testing network exploration
		import network_exploration

		results = network_exploration.run_cnn_exploration(train_X, train_label, 20, 10, 10, 2)
		fashion_model = network_exploration.craft_explored_model(input_size_x,
		                                                         results["nb_layers"],
																 2,
																 512,
																 results["nb_filters"],
																 results["dense_layer"],
																 results["dropout"],
																 num_classes)
		fashion_model.summary()

	else:
		## craft the default CNN architecture without optimisation

		## Neural network architecture
		fashion_model = Sequential()
		fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(input_size_x,input_size_y,1),padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D((2, 2),padding='same'))
		fashion_model.add(Dropout(0.25))

		fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Dropout(0.25))

		fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Dropout(0.4))

		fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Dropout(0.4))

		fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Dropout(0.4))

		fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Dropout(0.4))

		fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Dropout(0.4))

		fashion_model.add(Flatten())
		fashion_model.add(Dense(128, activation='linear'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(Dropout(0.3))
		fashion_model.add(Dense(num_classes, activation='softmax', name='preds'))

		## Compile the model
		fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
		fashion_model.summary()

	## Train the model
	fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

	## Save the model
	fashion_model.save("log/model.h5")

	## Compute AUC
	## work only for binary classification
	try:
		pred = fashion_model.predict_classes(test_X)
		fpr, tpr, thresholds = metrics.roc_curve(test_Y, pred)
		auc_value = metrics.auc(fpr, tpr)
	except:
		auc_value = "NA"

	## Evaluate the model
	test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
	print('Test loss:', test_eval[0])
	print('Test accuracy:', test_eval[1])
	print("Test AUC: "+str(auc_value))

	##-------------------------------------------------------##
	## Compute saliency and grad - CAM                       ##
	## Saliency, this corresponds to the Dense linear layer. ##
	## visualize activation over final dense layer outputs,  ##
	## we need to switch the softmax activation out for      ##
	## linear since gradient of output node will depend on   ##
	## all the other node activations.                       ##
	##-------------------------------------------------------##
	layer_idx = utils.find_layer_idx(fashion_model, 'preds')

	## Swap softmax with linear
	fashion_model.layers[layer_idx].activation = activations.linear
	model = utils.apply_modifications(fashion_model)

	## Compute the images
	for class_idx in numpy.arange(num_classes):

		indices = numpy.where(test_Y_one_hot[:, class_idx] == 1.)[0]
		idx = indices[0]

		f, ax = plt.subplots(1, 4)
		ax[0].imshow(test_X[idx][..., 0])

		for i, modifier in enumerate([None, 'guided', 'relu']):
			grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=test_X[idx], backprop_modifier=modifier)

			## Extract important pixels and their associated scores
			pos_to_score = gradient_visualisation.get_important_pixels(grads)

			## Extract feature importance and save information in a specific
			## log file
			gradient_visualisation.save_features_importance(grid_matrix,
															pos_to_score,
															modifier,
															class_idx,
															"saliency")
			if modifier is None:
				modifier = 'vanilla'

			ax[i+1].set_title(modifier)
			ax[i+1].imshow(grads, cmap='jet')

		plt.savefig("log/"+str(class_idx)+"_saliency.png")
		plt.close()

   	## grad - CAM
	for class_idx in numpy.arange(num_classes):
		indices = numpy.where(test_Y_one_hot[:, class_idx] == 1.)[0]
		idx = indices[0]

		f, ax = plt.subplots(1, 4)
		ax[0].imshow(test_X[idx][..., 0])

		for i, modifier in enumerate([None, 'guided', 'relu']):
			grads = visualize_cam(model, layer_idx, filter_indices=class_idx, seed_input=test_X[idx], backprop_modifier=modifier)

			## Extract important pixels and their associated scores
			pos_to_score = gradient_visualisation.get_important_pixels(grads)

			## Extract feature importance and save information in a specific
			## log file
			gradient_visualisation.save_features_importance(grid_matrix,
															pos_to_score,
															modifier,
															class_idx,
															"gradCam")
			if(modifier is None):
				modifier = 'vanilla'

			ax[i+1].set_title(modifier)
			ax[i+1].imshow(grads, cmap='jet')

		plt.savefig("log/"+str(class_idx)+"_grad_cam.png")
		plt.close()


   	##------------------------##
   	## Save and write results ##
   	##------------------------##

	## save results in log_file
	model_training_log_file = open("log/model_training.log", "w")
	model_training_log_file.write("test_accuracy;"+str(test_eval[1])+"\n")
	model_training_log_file.write("test_loss;"+str(test_eval[0])+"\n")
	model_training_log_file.write("test_auc;"+str(auc_value)+"\n")
	model_training_log_file.write("epochs;"+str(epochs)+"\n")
	model_training_log_file.write("observation_in_training;"+str(len(train_Y_one_hot))+"\n")
	model_training_log_file.write("observation_in_test;"+str(len(test_Y_one_hot))+"\n")
	model_training_log_file.close()

	## if the prediction dataset is not empty
	## use the trained network to predict
	## label on the prediction data,
	## write results on a csv file
	if(len(prediction_set) > 0):
		predicted_classes = fashion_model.predict(test_X)
		predicted_classes = numpy.argmax(numpy.round(predicted_classes),axis=1)
		prediction_results_file = open("prediction_results.csv", "w")
		cmpt = 0
		for prediction in predicted_classes:
			line_to_write = str(cmpt)+","+str(prediction)+"\n"
			prediction_results_file.write(line_to_write)
			cmpt += 1
		prediction_results_file.close()


	## Model evaluation, graphical output
	accuracy = fashion_train.history['acc']
	val_accuracy = fashion_train.history['val_acc']
	loss = fashion_train.history['loss']
	val_loss = fashion_train.history['val_loss']
	epochs = range(len(accuracy))
	plt.figure()
	plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
	plt.plot(epochs, val_accuracy, 'b', label='Test accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.savefig("log/validation.png")
	plt.close()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Test loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig("log/loss.png")
	plt.close()
	#plt.show()
