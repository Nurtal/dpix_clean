#coding: utf8
import matplotlib
matplotlib.use('TkAgg')
import os
import getopt
import sys


import representation
import preprocessing
import classification
import manager


def main(argv):
	"""
	Run the dpix module

	-a argument can be:
		- exemple : run an exemple
		- run : classic run on the specify dataset

	-> NB : strange error when not run as root
	"""

	## init action parameters
	action = "NA"

	## try to catch arguments
	try:
		opts, args = getopt.getopt(argv,"ha:",["action="])

	except getopt.GetoptError:
		sys.exit(2)

	## parse arguments
	for opt, arg in opts:

		## Display Help
		if opt == '-h':
			sys.exit()

		## Get action
		elif opt in ("-a", "--action"):
			action = arg


	## Run the exemple
	if(action == "exemple"):

		## Prepare to predict data - give an empty array
		prediction_dataset = []

		## Build image map and save it
		preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
		preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
		image_structure = representation.build_image_map_GMA("datasets/creditcard_reduce_reformated_scaled.csv", 100, False, False)
		manager.save_matrix_to_file(image_structure, "datasets/credit_image_structure.csv")

		## prepare train data
		representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
		representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)

		## Run CNN
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 200, prediction_dataset, image_structure)

		## write report
		manager.write_report()


	## Run the classic runner
	elif(action == "run"):

			## run solver on the target file
			## IN PROGRESS

			## parameters
			build_image_map_iteration = 5
			proximity_matrix = False

			## Prepare to predict data - give an empty array
			prediction_dataset = []

			## deal with file names
			target_file = argv[2]
			reformated_data_file_name = target_file.split(".")
			reformated_data_file_name = reformated_data_file_name[0]+"_reformated.csv"
			scaled_data_file_name = reformated_data_file_name.split(".")
			scaled_data_file_name = scaled_data_file_name[0]+"_scaled.csv"
			matrix_save_file_name = scaled_data_file_name.split(".")
			matrix_save_file_name = matrix_save_file_name[0]+"_saved_matrix.csv"
			interpolated_data_file_name = scaled_data_file_name.split(".")
			interpolated_data_file_name = interpolated_data_file_name[0]+"_interpolated.csv"

			## preprocessing
			last_pos = preprocessing.get_number_of_columns_in_csv_file(target_file) -1
			preprocessing.reformat_input_datasets(target_file, last_pos, True)
			preprocessing.normalize_data(reformated_data_file_name)

			## Build image structure
			image_structure = representation.build_image_map_GMA(scaled_data_file_name,
			 													 build_image_map_iteration,
																 proximity_matrix,
																 False)
			manager.save_matrix_to_file(image_structure, matrix_save_file_name)

			## prepare train data
			representation.simple_conversion_to_img_matrix(scaled_data_file_name)
			representation.build_patient_representation(interpolated_data_file_name, image_structure)
			real_data = representation.build_patient_matrix(interpolated_data_file_name, image_structure)
			(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.7)

			## Run CNN
			classification.run_CNN(train_X, train_Y, test_X, test_Y, 500, prediction_dataset, image_structure)

			## write report
			manager.write_report()




##------##
## MAIN ########################################################################
##------##


if __name__ == '__main__':
	main(sys.argv[1:])
