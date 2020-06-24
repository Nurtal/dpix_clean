##----------------------##
## Preprocessing module ##
##----------------------##
## pre process data file & generate random data
## for DPIX.
##
## MANIFEST:
##		-> generate_random_data
##		-> normalize_data
##		-> split_to_disease_vs_control
##		-> overclock_dataset
##		-> reformat_input_datasets
##		-> reformat_input_datasets
##		-> get_number_of_columns_in_csv_file
##
##


def normalize_data(data_file):
	##
	## -> Scale (centrer normer) the data
	## and write the data in new _scaled data
	## file
	##

	## importation
	from numpy import genfromtxt
	from sklearn import preprocessing

	## Get the header
	cmpt = 0
	header = ""
	input_data = open(data_file, "r")
	for line in input_data:
		line = line.replace("\n", "")
		if(cmpt == 0):
			header = line
		cmpt += 1
	input_data.close()

	## Get and scale data
	data = genfromtxt(data_file, delimiter=',')
	data_scaled = preprocessing.scale(data[1:])

	## Write new file
	output_file_name = data_file.split(".")
	output_file_name = output_file_name[0]+"_scaled.csv"

	output_data = open(output_file_name, "w")
	output_data.write(header+"\n")

	for vector in data_scaled:
		vector_to_write = ""
		for scalar in vector:
			vector_to_write += str(scalar)+","
		vector_to_write = vector_to_write[:-1]
		output_data.write(vector_to_write+"\n")
	output_data.close()



def split_to_disease_vs_control(data_file):
	"""
	Split precisesads data file using the Diagnosis column to create
	one data_file per register disease in the diagnosis column versus Control.

	-> remove the "N" letter in the OMICID id if it is present.

	-> assume disease column is labeled "Diagnosis"
	-> assume all control patients are labeled "Control"
	"""
	## importation
	import pandas as pd

	## parameters
	disease_to_data = {}
	output_prefix = data_file.split(".")
	output_prefix = output_prefix[0]+"_Control_vs_"

	## read data
	data = pd.read_csv(data_file)

	## deal with patient ID, remove "N" if it's present
	if(data["OMICID"][0][0] == "N"):
		data["OMICID"] = data["OMICID"].str.slice(1,10)

	disease_list = []
	for disease in data["Diagnosis"]:
		if(disease not in disease_list):
			disease_list.append(disease)

	## splitted dataset
	for disease in disease_list:
		disease_to_data[disease] = data[data["Diagnosis"] == disease]

	## merge with control dataset
	for disease in disease_to_data.keys():
		if(disease != "Control"):
			splited_dataset = pd.concat([disease_to_data[disease],
			                             disease_to_data["Control"]])
			splited_dataset.to_csv(output_prefix+str(disease)+".csv",
			                       index=False)




def overclock_dataset(input_dataset, classification_variable_position, increase_factor):
	"""
	IN PROGRESS

	Repeat n time (given by increase_factor) each variable in the dataset, use to create
	big images to process.
	"""

	## importation
	import pandas as pd

	## parameters
	output_file_name = input_dataset.split(".")
	output_file_name = output_file_name[0]+"_overclocked.csv"

	## read data, identify label variable
	dataset = pd.read_csv(input_dataset)
	class_label = dataset.keys()[classification_variable_position]
	class_data = dataset[class_label]

	## remove label variable
	dataset = dataset.drop(columns=[class_label])
	original_labels = dataset.keys()

	## increase dataset size
	for x in xrange(1,increase_factor):
		for label in original_labels:
			dataset[str(label)+"_clone_"+str(x)] = dataset[label]
		print("[+] Overclocking : "+str((float(x)/float(increase_factor))*100.0)+"% Completed")
	print("[+] Overclocking : "+str(100.0)+"% Completed")

	## re-add label to the dataset
	dataset[class_label] = class_data

	## save dataset in a file
	dataset.to_csv(output_file_name, index=False)


def reformat_input_datasets(input_dataset, classification_variable_position, force_square_matrix):
	##
	## IN TEST PHASE
	##
	## Reformat the input_dataset to be processed ny the PATREP functions
	## classification_variable_position is an integer, the position of
	## the classification variable in the header.
	##
	## force_square_matrix is a boolean, add "dead variables" to get
	## a square matrix in the end.
	##
	## Produce a manifest file to map the variables to their number
	## Produce a classification file to map the observations to teir class
	##
	## WARNING => Seems to bug when classification variable is in position 0
	##

	## importation
	import math

	if(force_square_matrix):

		## get the number of variables
		number_of_variables = -1
		input_data = open(input_dataset, "r")
		cmpt = 0
		for line in input_data:
			if(cmpt == 0):
				line_in_array = line.split(",")
				number_of_variables = len(line_in_array)-1
			cmpt += 1
		input_data.close()

		## test if we can build a square matrix with that
		test_value_1 = math.sqrt(number_of_variables)
		test_value_2 = int(test_value_1)
		optimal_number_of_variables = number_of_variables
		print("[DEBUG] => original number of variables : "+str(optimal_number_of_variables))
		if(float(test_value_1 - test_value_2) > 0):

			print("[DEBUG] => adapt number of variables")

			## find next cool number of variables
			can_do_something_with_this = False
			while(not can_do_something_with_this):

				test_value_1 = math.sqrt(optimal_number_of_variables)
				test_value_2 = int(test_value_1)
				if(float(test_value_1 - test_value_2) == 0):
					can_do_something_with_this = True
				else:
					optimal_number_of_variables	+= 1

		print("[DEBUG] => optimal number of variables : "+str(optimal_number_of_variables))

		## Create the index file for variables
		output_dataset_name = input_dataset.split(".")
		output_dataset_name = output_dataset_name[0]+"_reformated.csv"
		output_dataset = open(output_dataset_name, "w")
		input_dataset = open(input_dataset, "r")
		index_file = open("variable_manifest.csv", "w")
		classification_file = open("observations_classification.csv", "w")
		cmpt = 0

		## classification variables
		class_to_id = {}
		class_id = 0

		for line in input_dataset:
			line = line.replace("\n", "")

			## create new header
			## create index file
			if(cmpt ==0):
				new_header = ""
				line_in_array = line.split(",")
				index = 0
				last_real_variable_index = -1
				for variable in line_in_array:
					if(index != classification_variable_position):
						new_variable = "variable_"+str(index)
						last_real_variable_index = index
						new_header += str(new_variable)+","
						index_file.write(str(variable) +"," +str(new_variable)+"\n")
					index += 1

				## add dead variable to fit a square matrix
				index = last_real_variable_index + 1
				for x in range(optimal_number_of_variables - number_of_variables):

					## deal with the index += 1 from last loop
					new_variable = "variable_"+str(index)
					new_header += str(new_variable)+","
					index_file.write(str(variable) +"," +str(new_variable)+"\n")
					index += 1

				## Write the final header
				new_header = new_header[:-1]
				output_dataset.write(new_header+"\n")

			## fill output file
			## create classification file
			else:
				line_to_write = ""
				index = 0
				line_in_array = line.split(",")
				for scalar in line_in_array:
					if(index != classification_variable_position):
						line_to_write += str(scalar) + ","
					else:

						## associate an id (integer) to each class
						if(str(scalar) not in class_to_id.keys()):
							class_to_id[str(scalar)] = class_id
							class_id += 1

						## write in classification file
						classification_file.write(str(cmpt -1 ) + "," + str(scalar)+ ","+ str(class_to_id[str(scalar)])+"\n")
					index +=1

				## add dead variable to fit a square matrix
				for x in range(optimal_number_of_variables - number_of_variables):
					line_to_write += str(0)+","

				line_to_write = line_to_write[:-1]
				output_dataset.write(line_to_write+"\n")
			cmpt +=1

		classification_file.close()
		index_file.close()
		input_dataset.close()

	else:
		## Create the index file for variables
		output_dataset_name = input_dataset.split(".")
		output_dataset_name = output_dataset_name[0]+"_reformated.csv"
		output_dataset = open(output_dataset_name, "w")
		input_dataset = open(input_dataset, "r")
		index_file = open("variable_manifest.csv", "w")
		classification_file = open("observations_classification.csv", "w")

		## classification variables
		class_to_id = {}
		class_id = 0

		cmpt = 0
		for line in input_dataset:
			line = line.replace("\n", "")

			## create new header
			## create index file
			if(cmpt ==0):
				new_header = ""
				line_in_array = line.split(",")
				index = 0
				for variable in line_in_array:
					if(index != classification_variable_position):
						new_variable = "variable_"+str(index+1)
						new_header += str(new_variable)+","
						index_file.write(str(variable) +"," +str(new_variable)+"\n")
					index += 1
				new_header = new_header[:-1]
				output_dataset.write(new_header+"\n")

			## fill output file
			## create classification file
			else:
				line_to_write = ""
				index = 0
				line_in_array = line.split(",")
				for scalar in line_in_array:
					if(index != classification_variable_position):
						line_to_write += str(scalar) + ","
					else:

						## associate an id (integer) to each class
						if(str(scalar) not in class_to_id.keys()):
							class_to_id[str(scalar)] = class_id
							class_id += 1

						## write in classification file
						classification_file.write(str(cmpt -1 ) + "," + str(scalar)+ ","+ str(class_to_id[str(scalar)])+"\n")

					index +=1
				line_to_write = line_to_write[:-1]
				output_dataset.write(line_to_write+"\n")
			cmpt +=1

		classification_file.close()
		index_file.close()
		input_dataset.close()



def reformat_prediction_dataset(input_dataset):
	##
	## Based on the reformat_input_datasets function,
	## create a "_reformated" file for prediction dataset,
	## i.e no target column expected in the input file,
	## no manifest_variable and index file generated
	## in the process
	##

	## importation
	import math

	## get the number of variables
	number_of_variables = -1
	input_data = open(input_dataset, "r")
	cmpt = 0
	for line in input_data:
		if(cmpt == 0):
			line_in_array = line.split(",")
			number_of_variables = len(line_in_array)
		cmpt += 1
	input_data.close()

	## test if we can build a square matrix with that
	test_value_1 = math.sqrt(number_of_variables)
	test_value_2 = int(test_value_1)
	optimal_number_of_variables = number_of_variables
	if(float(test_value_1 - test_value_2) > 0):

		## find next cool number of variables
		can_do_something_with_this = False
		while(not can_do_something_with_this):

			test_value_1 = math.sqrt(optimal_number_of_variables)
			test_value_2 = int(test_value_1)
			if(float(test_value_1 - test_value_2) == 0):
				can_do_something_with_this = True
			else:
				optimal_number_of_variables	+= 1


	## Create output file
	output_dataset_name = input_dataset.split(".")
	output_dataset_name = output_dataset_name[0]+"_reformated.csv"
	output_dataset = open(output_dataset_name, "w")
	input_dataset = open(input_dataset, "r")

	cmpt = 0

	for line in input_dataset:
		line = line.replace("\n", "")

		## create new header
		## create index file
		if(cmpt ==0):
			new_header = ""
			line_in_array = line.split(",")
			index = 0
			last_real_variable_index = -1
			for variable in line_in_array:
				new_variable = "variable_"+str(index)
				last_real_variable_index = index
				new_header += str(new_variable)+","
				index += 1

			## add dead variable to fit a square matrix
			index = last_real_variable_index + 1
			for x in xrange(optimal_number_of_variables - number_of_variables):

				## deal with the index += 1 from last loop
				new_variable = "variable_"+str(index)
				new_header += str(new_variable)+","
				index += 1

			## Write the final header
			new_header = new_header[:-1]
			output_dataset.write(new_header+"\n")

		## fill output file
		else:
			line_to_write = ""
			index = 0
			line_in_array = line.split(",")
			for scalar in line_in_array:
				line_to_write += str(scalar) + ","
				index +=1

			## add dead variable to fit a square matrix
			for x in xrange(optimal_number_of_variables - number_of_variables):
				line_to_write += str(0)+","

			line_to_write = line_to_write[:-1]
			output_dataset.write(line_to_write+"\n")
		cmpt +=1

	output_dataset.close()
	input_dataset.close()



def get_number_of_columns_in_csv_file(csv_file):
	"""
	return the number of col
	in a csv file
	"""
	nb_col = "NA"
	data_file = open(csv_file, "r")
	for line in data_file:
		line = line.rstrip()
		line_in_array = line.split(",")
		nb_col = len(line_in_array)
	data_file.close()

	return nb_col
