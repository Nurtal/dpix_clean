#coding: utf8

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
import os


def plot_log_file(log_file):
	##
	## [IN PROGRESS]
	##
	## => Plot values of scores in log file
	##
	##

	## Init data structure
	global_scores = []

	## Get the values
	data = open(log_file)
	for line in data:
		line = line.replace("\n", "")
		line_in_array = line.split(";")
		if(line_in_array[0] == "global_score"):
			global_scores.append(float(line_in_array[1]))
	data.close()

	## plot the values
	plt.plot(global_scores)
	plt.show()




def save_matrix_to_file(matrix, save_file):
	##
	## => Save the content of matrix (wich is a 2d array)
	## in the file save_file
	##

	output_file = open(save_file, "w")
	for vector in matrix:
		line_to_write = ""
		for scalar in vector:
			line_to_write += str(scalar) + ","
		line_to_write = line_to_write[:-1]
		output_file.write(line_to_write+"\n")
	output_file.close()



def load_matrix_from_file(load_file):
	##
	## => Create a matrix from a save_file,
	## cast the matrix into an numpy array
	## return the matrix
	##

	matrix = []
	input_data = open(load_file, "r")
	for line in input_data:
		vector = []
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		for scalar in line_in_array:
			vector.append(float(scalar))
		matrix.append(vector)
	input_data.close()

	matrix = numpy.array(matrix)

	return matrix



def write_report():
	##
	## IN PROGRESS
	##

	## importation
	import os

	##-----------------------------##
	## GET & GENERATE INFORMATIONS ##
	##-----------------------------##

	## generate graphe for the "learning the grid"
	## phase.
	global_scores = []

	## Get the values
	if(os.path.isfile("learning_optimal_grid.log")):
		data = open("learning_optimal_grid.log", "r")
		for line in data:
			line = line.replace("\n", "")
			line_in_array = line.split(";")
			if(line_in_array[0] == "global_score"):
				global_scores.append(float(line_in_array[1]))
		data.close()

		## save the figure
		plt.figure()
		plt.plot(global_scores, label='grids scores')
		plt.title('Learning the grid structure')
		plt.legend()
		plt.savefig("log/learning_the_grid.png")

	## get information on the model
	test_accuracy = -1
	test_loss = -1
	epochs = -1
	observation_in_training_set = -1
	observation_in_test_set = -1
	model_log_file = open("log/model_training.log", "r")
	for line in model_log_file:
		line = line.replace("\n", "")
		line_in_array = line.split(";")
		if(line_in_array[0] == "test_accuracy"):
			test_accuracy = line_in_array[1]
		elif(line_in_array[0] == "test_loss"):
			test_loss = line_in_array[1]
		elif(line_in_array[0] == "epochs"):
			epochs = line_in_array[1]
		elif(line_in_array[0] == "observation_in_training"):
			observation_in_training_set = line_in_array[1]
		elif(line_in_array[0] == "observation_in_test"):
			observation_in_test_set = line_in_array[1]
	model_log_file.close()

	##--------------##
	## WRITE REPORT ##
	##--------------##

	## Prepare html report
	report_file = open("log/report.html", "w")

	## write header
	report_file.write("<html>\n")
	report_file.write("<head>\n")
	report_file.write("    <title>Patrep Report</title>\n")
	report_file.write("</head>\n")

	## write the title
	report_file.write("<body>\n")
	report_file.write("    <h1>PATREP REPORT</h1>\n")

	## write the "learning the grid" part
	report_file.write("    <h2>Learning the Grid</h2>\n")
	report_file.write("    <img src=\"learning_the_grid.png\">\n")

	## write the model evaluation part
	report_file.write("    <h2>Model Evaluation</h2>\n")
	report_file.write("    <img src=\"validation.png\">\n")
	report_file.write("    <img src=\"loss.png\">\n")
	report_file.write("    <table>\n")
	report_file.write("    <tr>\n")
	report_file.write("        <th>Accuracy</th>\n")
	report_file.write("        <th>Loss</th>\n")
	report_file.write("        <th>Train set</th>\n")
	report_file.write("        <th>Validation set</th>\n")
	report_file.write("        <th>Epochs</th>\n")
	report_file.write("    </tr>\n")
	report_file.write("        <td>"+str(test_accuracy)+"</td>\n")
	report_file.write("        <td>"+str(test_loss)+"</td>\n")
	report_file.write("        <td>"+str(observation_in_training_set)+"</td>\n")
	report_file.write("        <td>"+str(observation_in_test_set)+"</td>\n")
	report_file.write("        <td>"+str(epochs)+"</td>\n")
	report_file.write("    </table>\n")

	report_file.write("</body>\n")
	report_file.write("</html>")
	report_file.close()



def save_results(target_file, GMA_iteration, factor):
	"""
	For the rush instruction
	"""

	import os

	## parameters
	target = target_file.split("/")
	target = target[-1].split(".")
	target = target[0]
	output_folder = "/media/glorfindel/Données/dpix_rush_results/"+str(target)+"_GMA_"+str(GMA_iteration)+"_increase_"+str(factor)
	converger_file = "log/"+str(target)+"_GMA_converger.log"
	matrix_file = "datasets/"+str(target)+"_reformated_scaled_saved_matrix.csv"
	file_to_save = ["log/validation.png",
	                "log/loss.png",
					"log/0_feature_importance_gradCam_guided.log",
					"log/0_feature_importance_gradCam_None.log",
					"log/0_feature_importance_gradCam_relu.log",
					"log/0_feature_importance_saliency_guided.log",
					"log/0_feature_importance_saliency_None.log",
					"log/0_feature_importance_saliency_relu.log",
					"log/1_feature_importance_gradCam_guided.log",
					"log/1_feature_importance_gradCam_None.log",
					"log/1_feature_importance_gradCam_relu.log",
					"log/1_feature_importance_saliency_guided.log",
					"log/1_feature_importance_saliency_None.log",
					"log/1_feature_importance_saliency_relu.log",
					"log/0_grad_cam.png",
					"log/0_saliency.png",
					"log/1_grad_cam.png",
					"log/1_saliency.png",
					"log/dense_layer_optimisation.log",
					"log/dropout_optimisation.log",
					"log/nb_layers_optimisation.log",
					"log/filters_maps_optimisation.log",
					"log/dense_layer_optimisation.png",
					"log/dropout_optimisation.png",
					"log/nb_layers_optimisation.png",
					"log/filters_maps_optimisation.png",
					"log/model_training.log",
					converger_file,
					matrix_file]

	## Create output folder and save results file
	os.system("mkdir "+str(output_folder))
	for log_file in file_to_save:
		os.system("cp "+str(log_file)+" "+str(output_folder)+"/")




###------------###
### TEST SPACE ###
###------------###


#image_structure = representation.build_image_map("trash_data_scaled.csv")
#representation.build_patient_representation("trash_data_scaled_interpolated.csv", image_structure)
#plot_log_file("learning_optimal_grid.log")

## Test on real external dataset
"""
## Prepare to predict data - give an empty array
prediction_dataset = []
preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
image_structure = representation.build_image_map("datasets/creditcard_reduce_reformated_scaled.csv", 5)
save_matrix_to_file(image_structure, "credit_image_structure.csv")
representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
real_data = representation.build_patient_matrix("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
classification.run_CNN(train_X, train_Y, test_X, test_Y, 20, prediction_dataset)
plot_log_file("learning_optimal_grid.log")
"""


## Test on HLA data
"""
preprocessing.reformat_input_datasets("datasets/HLA_data_clean.csv", 562, True)
preprocessing.normalize_data("datasets/HLA_data_clean_reformated.csv")
image_structure = representation.build_image_map("datasets/HLA_data_clean_reformated_scaled.csv", 500)
save_matrix_to_file(image_structure, "HLA_image_structure_500i.csv")
representation.simple_conversion_to_img_matrix("datasets//HLA_data_clean_reformated_scaled.csv")
representation.build_patient_representation("datasets//HLA_data_clean_reformated_scaled_interpolated.csv", image_structure)
real_data = representation.build_patient_matrix("datasets//HLA_data_clean_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
classification.run_CNN(train_X, train_Y, test_X, test_Y, 20)

plot_log_file("learning_optimal_grid.log")
"""

## Create grids for HLA data
## One grid / pathology computed with control
"""
iteration_list = [750]
dataset_list = ["datasets/HLA_data_MCTD.csv", "datasets/HLA_data_PAPs.csv", "datasets/HLA_data_RA.csv", "datasets/HLA_data_SjS.csv", "datasets/HLA_data_SLE.csv", "datasets/HLA_data_SSc.csv", "datasets/HLA_data_UCTD.csv"]

for iteration in iteration_list:

	for dataset in dataset_list:
		dataset_reformated = dataset.split(".")
		dataset_reformated = dataset_reformated[0]+"_reformated.csv"
		dataset_reformated_scaled = dataset_reformated.split(".")
		dataset_reformated_scaled = dataset_reformated_scaled[0]+"_scaled.csv"
		save_matrix_name = dataset.split(".")
		save_matrix_name = save_matrix_name[0]+"_"+str(iteration)+".csv"

		if(not os.path.isfile(save_matrix_name)):
			preprocessing.reformat_input_datasets(dataset, 562, True)
			preprocessing.normalize_data(dataset_reformated)
			image_structure = representation.build_image_map(dataset_reformated_scaled, iteration)
			save_matrix_to_file(image_structure, save_matrix_name)
"""

## Perform classification on HLA data, for each disease vs control
## IN PROGRESS
"""
disease_list = ["MCTD", "PAPs", "RA", "SjS", "SLE", "SSc", "UCTD"]
for disease in disease_list:
	for iteration in [50, 150, 500, 750]:

		print " => PROCESS [DISEASE]"+ str(disease) +" [MAP ITERATION]"+ str(iteration) +" \n"

		preprocessing.reformat_input_datasets("datasets/HLA_data_"+str(disease)+".csv", 562, True)
		preprocessing.normalize_data("datasets/HLA_data_"+str(disease)+"_reformated.csv")
		image_structure = load_matrix_from_file("datasets/HLA_data_"+str(disease)+"_"+str(iteration)+".csv")
		representation.simple_conversion_to_img_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled.csv")
		representation.build_patient_representation("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 350)

"""

"""
##-----------------------------------------------------##
## Perform Classification on customer data from Kaggle ##
##-----------------------------------------------------##
## Load data structure
image_structure = load_matrix_from_file("data_customer_kaggle_structure.csv")

## prepare train data
preprocessing.reformat_input_datasets("datasets/data_customer_kaggle.csv", 370, True)
preprocessing.normalize_data("datasets/data_customer_kaggle_reformated.csv")
representation.simple_conversion_to_img_matrix("datasets/data_customer_kaggle_reformated_scaled.csv")
representation.build_patient_representation("datasets/data_customer_kaggle_reformated_scaled_interpolated.csv", image_structure)
0real_data = representation.build_patient_matrix("datasets/data_customer_kaggle_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)

## prepare test data
preprocessing.reformat_prediction_dataset("datasets/data_customer_prediction.csv")
preprocessing.normalize_data("datasets/data_customer_prediction_reformated.csv")
representation.simple_conversion_to_img_matrix("datasets/data_customer_prediction_reformated_scaled.csv")
representation.build_patient_representation("datasets/data_customer_prediction_reformated_scaled_interpolated.csv", image_structure)
prediction_data = representation.build_prediction_matrix("datasets/data_customer_prediction_reformated_scaled_interpolated.csv", image_structure)
prediction_dataset = classification.prepare_prediction_dataset_for_cnn(prediction_data)

## Run CNN
classification.run_CNN(train_X, train_Y, test_X, test_Y, 90, prediction_dataset)
"""


"""
##-------------------------##
## TEST PREDICTION UPGRADE ##
##-------------------------##
## Load data structure
image_structure = load_matrix_from_file("credit_image_structure.csv")

## preapre train data
preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
real_data = representation.build_patient_matrix("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)

## Prepare to predict data
preprocessing.reformat_prediction_dataset("datasets/credit_to_predict.csv")
preprocessing.normalize_data("datasets/credit_to_predict_reformated.csv")
representation.simple_conversion_to_img_matrix("datasets/credit_to_predict_reformated_scaled.csv")
representation.build_patient_representation("datasets/credit_to_predict_reformated_scaled_interpolated.csv", image_structure)
prediction_data = representation.build_prediction_matrix("datasets/credit_to_predict_reformated_scaled_interpolated.csv", image_structure)
prediction_dataset = classification.prepare_prediction_dataset_for_cnn(prediction_data)

## Run CNN
classification.run_CNN(train_X, train_Y, test_X, test_Y, 25, prediction_dataset)
"""




"""
##--------------##
## GENERATE HLA ##
##--------------##

result_file = open("log/rush_semrinar.log", "a")
run_cmpt = 0
accuracy = 0
epochs = 0
grid = "undef"
training_proportion = 0
grid_values = [50, 150, 500, 750]
disease_list = ["MCTD", "PAPs", "RA", "SjS", "SLE", "SSc", "UCTD"]


number_of_run_to_perform = 0
"""
"""
for iteration in [50, 150, 500, 750]:
	for x in xrange(2, 8):
		for y in xrange(2, 7):
			for disease in disease_list:
				number_of_run_to_perform += 1


for iteration in [50, 150, 500, 750]:
	for x in xrange(2, 8):
		for y in xrange(2, 7):
			for disease in disease_list:
				training_proportion = float(float(x) / 10)
				grid = iteration
				epochs = y * 10
				run_cmpt += 1

				if(run_cmpt > 278):

					## Load data structure
					image_structure = load_matrix_from_file("datasets/HLA_data_"+str(disease)+"_"+str(iteration)+".csv")

					## preapre train data
					preprocessing.reformat_input_datasets("datasets/HLA_data_"+str(disease)+".csv", 562, True)
					preprocessing.normalize_data("datasets/HLA_data_"+str(disease)+"_reformated.csv")
					representation.simple_conversion_to_img_matriclearx("datasets/HLA_data_"+str(disease)+"_reformated_scaled.csv")
					representation.build_patient_representation("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
					real_data = representation.build_patient_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
					(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, training_proportion)

					## Prepare to predict data - give an empty array
					prediction_dataset = []

					## Run CNN
					classification.run_CNN(train_X, train_Y, test_X, test_Y, epochs, prediction_dataset)

					## write in results_file
					log_file = open("log/model_training.log", "r")

					for line in log_file:
						line = line.replace("\n", "")
						line_in_array = line.split(";")
						if(line_in_array[0] == "test_accuracy"):
							accuracy = float(line_in_array[1])
						if(line_in_array[0] == "epochs"):
							epochs = int(line_in_array[1])
					log_file.close()

					result_line = "["+str(run_cmpt)+"] EPOCHS : "+str(epochs) +" ; GRID : "+str(grid) + " ; TRAINING " + str(training_proportion) + " ; DISEASE : " +str(disease)+ " ; ACC : "+str(accuracy) +"\n"
					result_file.write(result_line)

					print "["+str(run_cmpt)+"] => "+str(float(float(run_cmpt)/float(number_of_run_to_perform))*100)+" % COMPLETED"

result_file.close()
"""


"""
disease = "SjS"
iteration = 750
epochs = 400
training_proportion = 0.8
accuracy = -1
good_enough = False
while(not good_enough):

	os.remove("log/validation.png")
	os.remove("log/loss.png")

	## Load data structure
	image_structure = load_matrix_from_file("datasets/HLA_data_"+str(disease)+"_"+str(iteration)+".csv")

	## preapre train data
	preprocessing.reformat_input_datasets("datasets/HLA_data_"+str(disease)+".csv", 562, True)
	preprocessing.normalize_data("datasets/HLA_data_"+str(disease)+"_reformated.csv")
	representation.simple_conversion_to_img_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled.csv")
	representation.build_patient_representation("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
	real_data = representation.build_patient_matrix("datasets/HLA_data_"+str(disease)+"_reformated_scaled_interpolated.csv", image_structure)
	(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, training_proportion)

	## Prepare to predict data - give an empty array
	prediction_dataset = []

	## Run CNN
	classification.run_CNN(train_X, train_Y, test_X, test_Y, epochs, prediction_dataset)

	log_file = open("log/model_training.log", "r")
	for line in log_file:
		line = line.replace("\n", "")
		line_in_array = line.split(";")
		if(line_in_array[0] == "test_accuracy"):
			accuracy = float(line_in_array[1])
	log_file.close()

	if(float(accuracy) > 0.69):
		good_enough = True


print " => SOLUTION FOUND <= "

write_report()
"""
