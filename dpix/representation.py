##-----------------------##
## Representation module ##
##-----------------------##
## Represent observations as image.
## -> Learn the optimal structure for the image
##


def get_correlation_matrix(input_data_file):
	##
	## -> Compute and return the corrlation matrix
	## for the variables in input_data_file.
	## -> Assume every scalar could be cast to float
	## -> the variables in the correlation matrix are in
	## the same order as the variables in the header
	##
	##

	## importation
	import numpy

	## parameters
	variables_to_values = {}
	index_to_variables = {}

	## get data
	input_data = open(input_data_file, "r")
	cmpt = 0
	for line in input_data:
		line = line.replace("\n", "")

		## get header
		if(cmpt == 0):
			line_in_array = line.split(",")
			index = 0
			for variable in line_in_array:
				index_to_variables[index] = variable
				variables_to_values[variable] = []
				index += 1

		## get scalar
		else:
			line_in_array = line.split(",")
			index = 0
			for scalar in line_in_array:
				variables_to_values[index_to_variables[index]].append(float(scalar))
				index += 1

		cmpt += 1
	input_data.close

	## Compute correlation matrix
	variables_matrix = []
	index = 0
	for key in variables_to_values.keys():
		variables_matrix.append([])
		variables_matrix[index] = variables_to_values[index_to_variables[index]]
		index += 1

	## warning : generate NaN when forced squared Matrix
	correlation_matrix = numpy.corrcoef(variables_matrix)

	## Replace NaN by 0
	correlation_matrix = numpy.where(numpy.isnan(correlation_matrix), numpy.ma.array(correlation_matrix, mask=numpy.isnan(correlation_matrix)).mean(axis=0), correlation_matrix)

	return correlation_matrix




def reformat_bibot_proximity_matrix(matrix_file):
	"""
	=> Load matrix computed using bibot, add dummy variable if needed to generate
	a square matrix (columns and row) by scanning the variable manifest file.
	-> Overwrite the matrix_file with the new formated matrix.
	"""

	## importation
	import pandas as pd

	## parameters
	number_of_variables = -1
	variable_to_add = []
	row_to_add = []

	## drop unnamed columns (usually first column)
	dataset = pd.read_csv(matrix_file)
	dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

	## get number of variable to add
	#-> get the number of variables in the matrix files
	number_of_variables = len(dataset.keys())

	#-> get number and names of variables in the manifest file
	cmpt = 0
	manifest_data = open("variable_manifest.csv", "r")
	for line in manifest_data:
		cmpt +=1
		line = line.rstrip()
		line_in_array = line.split(",")
		if(cmpt > number_of_variables):
			variable_to_add.append(line_in_array[1])
	manifest_data.close()

	## check if we need to add variables in the matrix
	if(len(variable_to_add) > 0):

		## add columns
		for variable in variable_to_add:
			dataset[str(variable)] = 0

		## add rows
		data_len = len(dataset.keys())
		for variable in variable_to_add:
			values_serie = []
			## create values
			for x in range(0, data_len):
				values_serie.append(0)
			row_to_add.append(values_serie)

		df2 = pd.DataFrame(row_to_add, columns=dataset.keys())
		dataset = dataset.append(df2)

	## write the final datasets
	dataset.to_csv(matrix_file, index=False)










def create_image_from_csv(data_file, image_file):
	##
	## -> Create an image from a csv file,
	## - the image have to be png
	## - the values in data_file have to be integer between 0 and 255
	## - drop the header in data_file
	##

	## importation
	from numpy import genfromtxt
	import png

	data = genfromtxt(data_file, delimiter=',')
	data = data[1:] ## drop the header
	matrix = []
	for vector in data:
		matrix.append(list(vector))

	## save the image
	png.from_array(matrix, 'L').save(image_file)



def simple_conversion_to_img_matrix(data_file):
	##
	## -> very basic conversion from
	##  normalized data to a 0 - 255 ranged
	##  values.
	##
	## -> for each variables get the min
	## and max values, then map each scalar
	## of the variables to [0-255]
	##
	## => TODO:
	##	- find a cool name for the function
	##

	## importation
	from scipy.interpolate import interp1d

	## init structures
	variables_to_maxmin = {}
	index_to_variables = {}
	variables_to_values = {}
	variables_to_interpolatevalues = {}

	## Read input data
	## - fill the structures
	## - find min and max values for each variables
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:

		line = line.replace("\n", "")

		## Parse the header
		if(cmpt == 0):

			line_in_array = line.split(",")
			index = 0
			for variable in line_in_array:
				index_to_variables[index] = variable
				variables_to_maxmin[variable] = {"max":-765, "min":765}
				variables_to_values[variable] = []
				variables_to_interpolatevalues[variable] = []
				index +=1


		## - Get max value for each variables
		## - Get min value for each variables
		else:

			line_in_array = line.split(",")
			index = 0
			for scalar in line_in_array:
				variable = index_to_variables[index]
				variables_to_values[variable].append(float(scalar))
				variable_max = variables_to_maxmin[variable]["max"]
				variable_min = variables_to_maxmin[variable]["min"]
				if(float(scalar) > float(variable_max)):
					variables_to_maxmin[variable]["max"] = float(scalar)
				if(float(scalar) < float(variable_min)):
					variables_to_maxmin[variable]["min"] = float(scalar)
				index +=1
		cmpt += 1
	input_data.close()


	## Map each variables to [0-255]
	## Use interpol1d from Scipy
	for variable in variables_to_values.keys():
		max_val = variables_to_maxmin[variable]["max"]
		min_val = variables_to_maxmin[variable]["min"]
		interpolation = interp1d([min_val,max_val],[0,255])
		number_of_patients = 0
		for scalar in variables_to_values[variable]:
			scalar_interpolated = interpolation(float(scalar))
			scalar_interpolated = int(scalar_interpolated)
			variables_to_interpolatevalues[variable].append(scalar_interpolated)
			number_of_patients += 1

	## Write new file with interpolated data
	output_file_name = data_file.split(".")
	output_file_name = output_file_name[0]+"_interpolated.csv"

	output_file = open(output_file_name, "w")

	## Write the header
	header = ""
	for variable in variables_to_interpolatevalues.keys():
		header += str(variable)+","
	header = header[:-1]
	output_file.write(header+"\n")

	for x in range(0, number_of_patients):
		line_to_write = ""
		for variable in variables_to_interpolatevalues.keys():
			vector = variables_to_interpolatevalues[variable]
			line_to_write += str(vector[x])+","
		line_to_write = line_to_write[:-1]
		output_file.write(line_to_write+"\n")

	output_file.close()






def init_grid_matrix(corr_mat):
	##
	## Just a function to instanciate
	## randomly a grid matrix
	##

	## importation
	import numpy
	import math
	import random

	number_of_variables = len(corr_mat[0])
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)


	## Randomy assign a position in the grid to a variable
	variable_to_position = {}
	cmpt_assigned = 0 # debug
	for x in range(0,number_of_variables):

		position_assigned = False

		while(not position_assigned):
			x_position = random.randint(0,side_size-1)
			y_position = random.randint(0,side_size-1)
			position = [x_position, y_position]

			print(position)

			if(position not in variable_to_position.values()):
				variable_to_position[x] = position
				position_assigned = True

				cmpt_assigned += 1 # debug
				print("=> " +str(cmpt_assigned) +" position assigned ["+str(float(float(cmpt_assigned)/float(number_of_variables))*100) +"%]")

	## Write a matrix with the assigned position as coordinates
	## for the variables

	## init the matrix
	## [WARNINGS] => Still assume we deal
	## with a square matrix / image for the representation
	## of the patient

	## Init matrix
	map_matrix = numpy.zeros(shape=(side_size,side_size))

	## Fill the matrix
	for variable in variable_to_position.keys():
		position = variable_to_position[variable]
		x_position = position[0]
		y_position = position[1]

		map_matrix[x_position][y_position] = variable


	return map_matrix





def get_neighbour(var, grid_matrix, half_size_block):
	##
	## Get and return the list of neighbour of var in grid_matrix
	##
	## -> half_size_block is an integer, determine the size of the block of
	## neighbour around var.
	##


	## locate var in grid
	x_position = -1
	y_position = -1

	row = 0
	for x in grid_matrix:
		column = 0
		for y in x:
			if(y == var):
				x_position = row
				y_position = column
			column += 1
		row += 1

	## Get the neighbour
	neighbours = []


	##  _____
	## |-----|
	## |-----|
	## |+++--|
	## |+++--|
	## |+++--|
	## |_____|
	##
	if(y_position >= half_size_block and x_position >= half_size_block):
		for x in range(0,-half_size_block):
			n_x_position = x
			for y in range(0,-half_size_block):
				n_y_position = y
				n = grid_matrix[n_x_position][n_y_position]

				if(n_x_position != x_position or n_y_position != y_position):
					if(n not in neighbours):
						neighbours.append(n)


	##  _____
	## |-----|
	## |-----|
	## |--+++|
	## |--+++|
	## |--+++|
	## |_____|
	##
	if(y_position >= half_size_block and x_position <= len(grid_matrix[0])-half_size_block):
		for x in range(0, half_size_block):
			n_x_position = x
			for y in range(0,-half_size_block):
				n_y_position = y
				n = grid_matrix[n_x_position][n_y_position]

				if(n_x_position != x_position or n_y_position != y_position):
					if(n not in neighbours):
						neighbours.append(n)


	##  _____
	## |+++--|
	## |+++--|
	## |+++--|
	## |-----|
	## |-----|
	## |_____|
	##
	if(y_position <= len(grid_matrix[0]) - half_size_block and x_position >= half_size_block):
		for x in range(0,-half_size_block):
			n_x_position = x
			for y in range(0,half_size_block):
				n_y_position = y
				n = grid_matrix[n_x_position][n_y_position]

				if(n_x_position != x_position or n_y_position != y_position):
					if(n not in neighbours):
						neighbours.append(n)

	##  _____
	## |--+++|
	## |--+++|
	## |--+++|
	## |-----|
	## |-----|
	## |_____|
	##
	if(y_position <= len(grid_matrix[0]) - half_size_block and x_position <= len(grid_matrix[0]) - half_size_block):
		for x in range(0,half_size_block):
			n_x_position = x
			for y in range(0,half_size_block):
				n_y_position = y
				n = grid_matrix[n_x_position][n_y_position]

				if(n_x_position != x_position or n_y_position != y_position):
					if(n not in neighbours):
						neighbours.append(n)


	## return the list of neighbours
	return neighbours




def get_neighbours_by_var_id(var, grid_matrix):
	##
	## get the list of neighbour of var in grid_matrix
	##

	## locate var in grid
	x_position = -1
	y_position = -1

	row = 0
	for x in grid_matrix:
		column = 0
		for y in x:
			if(y == var):
				x_position = row
				y_position = column
			column += 1
		row += 1

	## Get the neighbour
	neighbours = []

	if(y_position > 0):
		n_x_position = x_position
		n_y_position = y_position - 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position < len(grid_matrix[0])-1):
		n_x_position = x_position
		n_y_position = y_position + 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(x_position > 0):
		n_x_position = x_position - 1
		n_y_position = y_position
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(x_position < len(grid_matrix)-1):
		n_x_position = x_position + 1
		n_y_position = y_position
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)

	if(y_position > 0 and x_position > 0):
		n_x_position = x_position - 1
		n_y_position = y_position - 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position > 0 and x_position < len(grid_matrix)-1):
		n_x_position = x_position + 1
		n_y_position = y_position - 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position < len(grid_matrix[0])-1 and x_position > 0):
		n_x_position = x_position - 1
		n_y_position = y_position + 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)
	if(y_position < len(grid_matrix[0]) - 1 and x_position < len(grid_matrix) - 1):
		n_x_position = x_position + 1
		n_y_position = y_position + 1
		n = grid_matrix[n_x_position][n_y_position]
		neighbours.append(n)

	## return the list of neighbours
	return  neighbours




def compute_matrix_score(tuple_stuff):
	##
	## Score is the sum of distance between each element of the grid
	## and its neighbour
	##
	## test for multiprocessing

	## absolute value

	corr_matrix = tuple_stuff[0]
	grid_matrix = tuple_stuff[1]

	corr_matrix = abs(corr_matrix)

	total_score = 0.0

	## compute half size block value, must be an integer
	half_size_block = int(len(grid_matrix) / 2)

	## for each pixel:
	for vector in grid_matrix:
		for scalar in vector:

			## get neighbour
			neighbours = get_neighbour(scalar, grid_matrix, half_size_block)

			## compute score for each pixel (the distance from each variable with their neighbours)
			for n in neighbours:
				total_score += corr_matrix[int(scalar)][int(n)]


	## return global score
	return total_score


def compute_grid_score_wrapper(corr_matrix, grid_matrix):
	"""
	Split the grid matrix and use compute matrix score
	function on each splitted block using multiprocessing

	-> x 10 acceleration on 119*119 grid compared to linear
	approach

	-> return the grid score (actually sum of the splitted grid's score)
	"""


	## importation
	import numpy as np
	from pathos.multiprocessing import ProcessingPool as Pool
	from operator import add
	import functools

	## split the grid
	half_split = np.array_split(grid_matrix, 2)
	res = map(lambda x: np.array_split(x, 2, axis=1), half_split)

	## python 2
	#splitted_grids = reduce(add, res)

	## python 3
	splitted_grids = functools.reduce(add, res)

	## Multiprocessing, call score function on each
	## part of the splitted grid
	score = 0
	inputs = []

	## create inputs
	for grid in splitted_grids:
		inputs.append((corr_matrix, grid))

	## run process
	try:
		res = Pool().amap(compute_matrix_score, inputs)
		results = res.get()
	except:

		## DEBUG : pool lib mihgt break memory on big grids
		## trying to split the inputs to solve the problem
		print("[WARNINGS] : Grid dcore computint : BREAKING MEMORY")
		print("[WARNINGS] : Attempt to reduce charge on CPU : splitting multiprocessing")

		inputs_1 = inputs[0:int(len(inputs)/2)]
		inputs_2 = inputs[int(len(inputs)/2):]

		res_1 = Pool().amap(compute_matrix_score, inputs_1)
		results_1 = res.get()

		res_2 = Pool().amap(compute_matrix_score, inputs_2)
		results_2 = res.get()

		results = results_1 + results_2


	## compute results
	for scalar in results:
		score += scalar

	return score




def select_random_psition_from_grid(grid):
    """
	-> Part of the GMA algorithm, used in score_converger.
    -> Select a random position within the grid and
       return the selected coordinates and the
       corresponding values in the grid in a tuple:
            (y,x,value)
    """

    ## importation
    import random

    ## set limits
    x_size = len(grid[0])
    y_size = len(grid)

    ## generate random coordinates
    x_coord = random.randint(0,x_size-1)
    y_coord = random.randint(0,y_size-1)

    ## find associated target
    target = grid[y_coord][x_coord]

    return (y_coord,x_coord,target)


def compute_potential_score(var_id,neighbours,dist_matrix):
    """
	-> Part of the GMA algorithm, used in score_converger.
    compute the potential score of a block (var_id and it's potential
    surrounding giving by the neighbours list) using the dist_matrix
    """

    potential_score = 0.0
    for n in neighbours:
        potential_score += dist_matrix[int(var_id)][int(n)]

    return potential_score




def score_converger(grid_matrix, distance_matrix):
    """
    -> Important part of the GMA algorithm
    """
    ## -> compute local score funtion
    ##      --> take the central variable and it's direct environement as
    ##          arguments
    ## -> update grid function
    ##

    ## importation
    import operator

    ## parameters
    number_of_candidate = int(len(distance_matrix)/4)
    max_number_of_candidate = 500
    min_number_of_candidate = 2
    candidate_list = {}
    number_of_proposition = number_of_candidate
    proposition_list = {}
    distance_matrix = abs(distance_matrix)
    proposition_assignement = {}
    candidate_assignement = {}


    ## Adapt number of candidates
    if(number_of_candidate > max_number_of_candidate):
        number_of_candidate = max_number_of_candidate
    elif(number_of_candidate < min_number_of_candidate):
        number_of_candidate = min_number_of_candidate

    ## select list of random candidate position in the grid
    ## carreful with the potential infinite loop
    while(len(candidate_list.keys()) < number_of_candidate):
        random_position = select_random_psition_from_grid(grid_matrix)
        if(random_position[2] not in candidate_list.keys()):
            candidate_list[random_position[2]] = random_position

    ## select list of random proposition position in the grid
    ## carreful with the potential infinite loop
    while(len(proposition_list.keys()) < number_of_proposition):
        random_position = select_random_psition_from_grid(grid_matrix)
        if(random_position[2] not in candidate_list.keys() and random_position[2] not in proposition_list.keys()):
            proposition_list[random_position[2]] = random_position

    ## compute local score for each candidate and proposition position
    ## -> candidate position
    candidate_to_score = {}
    for elt in candidate_list.keys():
        var_id = elt
        environement = get_neighbours_by_var_id(var_id, grid_matrix)
        actual_score = compute_potential_score(var_id,environement,distance_matrix)
        candidate_to_score[var_id] = actual_score

    ## -> proposition position
    proposition_to_score = {}
    for elt in proposition_list.keys():
        var_id = elt
        environement = get_neighbours_by_var_id(var_id, grid_matrix)
        potential_score = compute_potential_score(var_id,environement,distance_matrix)
        proposition_to_score[var_id] = potential_score

    ## comparison & decision
    ## -> split best and worst scores between candidates
    sorted_candidate = sorted(candidate_to_score.items(), key=operator.itemgetter(1))
    moving_candidates = sorted_candidate[0:int(len(sorted_candidate)/2)]
    for m in moving_candidates:
        candidate_assignement[m[0]] = False


    ## init proposition assignment
    sorted_proposition_candidates = sorted(proposition_to_score.items(), key=operator.itemgetter(1))
    for p in sorted_proposition_candidates:
        proposition_assignement[p[0]] = False

    ## -> for candidate in worst candidates see if an exchange with a
    ## proposition position is beneficial, process to exhange if i'ts the case.
    for c in moving_candidates:
        var_id = c[0]
        var_actual_score = c[1]

        for p in sorted_proposition_candidates:
            p_id = p[0]
            p_score = p[1]

            if(not proposition_assignement[p_id] and not candidate_assignement[var_id]):

                ## compute potential score
                var_potential_env = get_neighbours_by_var_id(p_id, grid_matrix)
                prop_potential_env = get_neighbours_by_var_id(var_id, grid_matrix)
                var_potential_score = compute_potential_score(var_id,var_potential_env,distance_matrix)
                prop_potential_score = compute_potential_score(p_id,prop_potential_env,distance_matrix)

                ## take decision
                if(var_potential_score > var_actual_score and var_potential_score > p_score and prop_potential_score > var_actual_score):

                    c_x = candidate_list[var_id][1]
                    c_y = candidate_list[var_id][0]

                    p_x = proposition_list[p_id][1]
                    p_y = proposition_list[p_id][0]

                    ## write gain in log file
                    log_file = open("converger.log", "a")
                    gain = float(var_potential_score - p_score) + float(prop_potential_score - var_actual_score)
                    print("[+] Change "+str(var_id) +" to "+str(p_id) +" => + "+str(gain))
                    log_file.write("[+] Change "+str(var_id) +" to "+str(p_id) +" => + "+str(gain)+"\n")
                    log_file.close()

                    ## Update the grid
                    grid_matrix[c_y][c_x] = p_id
                    grid_matrix[p_y][p_x] = var_id

                    proposition_assignement[p_id] = True
                    candidate_assignement[var_id] = True


    ## return updated grid
    return grid_matrix



def build_image_map_GMA(scaled_data_file, nb_iteration, proximity_matrix, scan_start):
	"""
	-> Use the Guided-Mutation-Algorithm to build a grid from a reformated
	and scaled data file, design as a replacement for the classic genetic
	mutation algorithm based function : build_image_map.

	-> parameters:
		- scaled_data_file is a string, the name of a scaled and reformated
		data file to build the grid
		- nb_iteration is an int, number of iteration for the algorithm
		- proximity_matrix is a the name of the
		proximity matrix file, can be set to FALSE to compute the
		only the correlation matrix and use it as the distance matrix
		("old fashion") but ususally use grid generated by BIBOT when
		using biological data.
		- [NEW / IN PROGRESS] scan_start is boolean, when set to True the
		function scan the grids folder, looking for already computed grids
		and start the computation from there. Open log file in append mod,
		search for second last saved iterations and pick up the computation
		from there.
	-> return a grid (save the grid and all the intermediate grids in the grids
       folder)
	-> Important sub-parameters:
		- steps is set by default to 10, represent the number of iteration
		performed before evalute the grid.
		- number of candidates is computed from the grid len in the
		score_converger function (by default max is 500, min is 2)
	-> Use multi-processing to compute the grid through the
	compute_grid_score_wrapper function.

	"""
	## importation
	import datetime
	import manager
	import os.path
	import pandas as pd
	import glob

	## parameters
	iterations = nb_iteration
	steps = 10
	log_file_name = scaled_data_file.split("/")
	log_file_name = log_file_name[-1].split(".")
	log_file_name = log_file_name[0].replace("_reformated_scaled", "")
	grid_file_name = log_file_name
	log_file_name = "log/"+log_file_name+"_GMA_converger.log"
	grid_file_name = "grids/"+grid_file_name+"_grid.csv"
	clear_to_pick_up = False

	## build first image
	print("[+] Creating first image ...")

	## default proximity matrix is the correlation matrix

	## try to reach proximity_matrix parameters
	if(proximity_matrix):
		if(os.path.isfile(str(proximity_matrix))):
			reformat_bibot_proximity_matrix(proximity_matrix)
			corr_matrix = pd.read_csv(proximity_matrix)
			corr_matrix = corr_matrix.as_matrix()
		else:
			## display waring
			display_m = "[PROXIMITY COMPUTATION][WARNING] => unable to use "
			display_m += str(proximity_matrix)+", unvalid file name"
			print(display_m)

			## default proximity matrix is the correlation matrix
			corr_matrix = get_correlation_matrix(scaled_data_file)
			## display final info
			display_m = "[PROXIMITY COMPUTATION][WARNING] => switching to "
			display_m += "strict correlation as proximity values"
			print(display_m)


	else:
		## default proximity matrix is the correlation matrix
		corr_matrix = get_correlation_matrix(scaled_data_file)


	## Check if it's a fresh run or a continuation
	if(scan_start and os.path.isfile(str(log_file_name))):

		## check if the log_file already exist
		if(os.path.isfile(str(log_file_name))):

			## Looking for existing grids in the grid folder
			## and keep the second higher iteration, not the first because
			## it might be corrupt by the event that stop the precedent run
			## a power failure for exemple ...
			save_file = grid_file_name.split(".")
			grids_file = glob.glob(save_file[0]+"*.csv")
			best_iteration = 0
			for gf in grids_file:
				it = gf.split("_")
				it = it[-1]
				it = it.replace(".csv", "")
				try:
					it = int(it)
				except:
					it = 0
				if(int(it) > best_iteration):
					best_iteration = it

			## check if target grid exist
			best_iteration = best_iteration - steps
			target_grid = save_file[0]+"_iteration_"+str(best_iteration)+".csv"
			if(os.path.isfile(str(target_grid))):

				## test if there is iteration to perform
				## if true:
				## - set the clear_to_pick_up variable to True
				## - open log file with the append mod
				## load grid_matrix from the target grid file
				if(iterations - best_iteration > 0):
					clear_to_pick_up = True
					iterations = iterations - best_iteration
					global_log_file = open(log_file_name, "a")
					grid_matrix = pd.read_csv(target_grid)
					grid_matrix = grid_matrix.as_matrix()
					start_iteration = best_iteration
				else:
					warning_m = "[PROXIMITY COMPUTATION][PICK UP][WARNING] => "
					warning_m += "no iteration to perform"
					print(warning_m)
					grid_matrix = init_grid_matrix(corr_matrix)
					global_log_file = open(log_file_name, "w")
					start_iteration = 0

			else:
				warning_m = "[PROXIMITY COMPUTATION][PICK UP][WARNING] => "
				warning_m += "Failed to load "+str(target_grid)
				print(warning_m)
				grid_matrix = init_grid_matrix(corr_matrix)
				global_log_file = open(log_file_name, "w")
				start_iteration = 0
		else:
			warning_m = "[PROXIMITY COMPUTATION][PICK UP][WARNING] => Failed "
			warning_m += "to pick up "+str(log_file_name)
			print(warning_m)
			grid_matrix = init_grid_matrix(corr_matrix)
			global_log_file = open(log_file_name, "w")
			start_iteration = 0
	else:
		grid_matrix = init_grid_matrix(corr_matrix)
		global_log_file = open(log_file_name, "w")
		start_iteration = 0



	## START GMA
	print("[GMA][START] => "+str(datetime.datetime.now()))

	cmpt = 0
	for x in range(start_iteration,iterations):

		## update the cursor x to the iteration from where we pick up
		## the computation
		if(clear_to_pick_up and x == 0):
			status_m = "[PROXIMITY COMPUTATION][PICK UP][SUCCESS] => pick up at"
			status_m += " iteration "+str(best_iteration)
			x = best_iteration

		grid_matrix = score_converger(grid_matrix, corr_matrix)
		if(cmpt == steps):
			print("[*] ITREATION "+str(x) +" => Computing grid score ...")
			score = compute_grid_score_wrapper(corr_matrix, grid_matrix)
			print("[*] ITREATION "+str(x) +" => SCORE : "+str(score))
			global_log_file.write(str(x)+","+str(score)+"\n")

			## save matrix
			save_file = grid_file_name.split(".")
			save_file = save_file[0]+"_iteration_"+str(x)+".csv"
			manager.save_matrix_to_file(grid_matrix, save_file)

			cmpt = 0
		cmpt += 1

	print("[GMA][END] => "+str(datetime.datetime.now()))
	manager.save_matrix_to_file(grid_matrix, grid_file_name)
	global_log_file.close()

	return grid_matrix





def build_patient_representation(data_file, map_matrix):
	##
	## Generate an image for each observation (line, except header)
	## in data_file.
	## map_matrix is the structure of the image to generate
	## all images are generated in the images folder.
	##

	## importation
	import numpy
	import math

	## get cohorte data
	index_to_variable = {}
	cohorte = []
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:
		patient = {}
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if(cmpt == 0):
			index = 0
			for variable in line_in_array:
				index_to_variable[index] = variable
				index += 1
		else:
			index = 0
			for scalar in line_in_array:
				patient[index_to_variable[index]] = float(scalar)
				index += 1
			cohorte.append(patient)
		cmpt +=1
	input_data.close()


	## get map for the image structure
	variable_to_position = {}
	for x in range(0,len(map_matrix)):
		vector = map_matrix[x]
		for y in range(0,len(vector)):
			variable = map_matrix[x][y]
			position = [x,y]
			variable_to_position[variable] = position


	## create image for each patients in cohorte
	number_of_variables = len(index_to_variable.keys())
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)
	cmpt = 0
	for patient in cohorte:

		## init patient grid
		patient_grid = numpy.zeros(shape=(side_size,side_size))

		## fill the patient grid
		for variable in index_to_variable.values():

			## get the variable id from information in the header
			variable_id = variable.split("_")
			variable_id = float(variable_id[-1])

			## get variable position in the grid
			position = variable_to_position[variable_id]
			position_x = position[0]
			position_y = position[1]

			## assign corresponding value to the position
			patient_grid[position_x][position_y] = int(patient[variable])

		## create the image dor the patient
		## write csv file for the patient

		csv_file = open("image_generation.tmp", "w")

		## deal with header
		header = ""
		for x in range(0, len(patient_grid[0])):
			header += str(x)+","
		header = header[:-1]
		csv_file.write(header+"\n")

		## write data
		for vector in patient_grid:
			line_to_write = ""
			for scalar in vector:
				line_to_write += str(scalar) + ","
			line_to_write = line_to_write[:-1]
			csv_file.write(line_to_write+"\n")
		csv_file.close()

		## generate image from csv file
		image_file = "images/patient_"+str(cmpt)+".png"
		create_image_from_csv("image_generation.tmp", image_file)

		cmpt += 1




def build_patient_matrix(data_file, map_matrix):
	##
	## data_file is usualy the reduce formated scaled interpolated version
	## of the input data.
	##
	## map_matrix is the learned structure of the image.
	##
	## return the structure: {patient_id:[matrix,diagnostic]}
	## where matrix is a representation of the image associated to
	## the patient.
	##

	## importation
	import numpy
	import math

	## Init variables
	data_structure = {}

	## get cohorte data
	index_to_variable = {}
	cohorte = []
	input_data = open(data_file, "r")
	cmpt = 0

	for line in input_data:
		patient = {}
		patient_id = cmpt
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if(cmpt == 0):
			index = 0
			for variable in line_in_array:
				index_to_variable[index] = variable
				index += 1
		else:
			index = 0
			for scalar in line_in_array:
				patient[index_to_variable[index]] = int(scalar)
				index += 1
			cohorte.append(patient)
			data_structure[patient_id] = ["choucroute", "choucroute"]
		cmpt +=1
	input_data.close()


	## get map for the image structure
	variable_to_position = {}
	for x in range(0,len(map_matrix)):
		vector = map_matrix[x]
		for y in range(0,len(vector)):
			variable = map_matrix[x][y]
			position = [x,y]
			variable_to_position[variable] = position

	## create matrix for each patients in cohorte
	number_of_variables = len(index_to_variable.keys())
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)
	cmpt = 0
	cohorte_matrix = []
	for patient in cohorte:

		## init patient grid
		patient_grid = numpy.zeros(shape=(side_size,side_size))
		patient_id = cmpt + 1

		## fill the patient grid
		for variable in index_to_variable.values():

			## get the variable id from information in the header
			variable_id = variable.split("_")
			variable_id = float(variable_id[-1])

			## get variable position in the grid
			position = variable_to_position[variable_id]
			position_x = position[0]
			position_y = position[1]

			## assign corresponding value to the position
			patient_grid[position_x][position_y] = int(patient[variable])

		## fill data structure
		data_structure[patient_id][0] = patient_grid
		patient_manifest_file = open("observations_classification.csv", "r")
		for line in patient_manifest_file:
			line = line.replace("\n", "")
			line_in_array = line.split(",")

			p_id = line_in_array[0]
			p_diag = line_in_array[1]
			p_diag_id = line_in_array[2]

			if(patient_id == int(p_id)+1):
				p_diag = p_diag.replace("\"", "")
				data_structure[patient_id][1] = int(p_diag_id)

		patient_manifest_file.close()
		cmpt += 1

	return data_structure




def build_prediction_matrix(data_file, map_matrix):
	##
	## Adapt from build_patient_matrix for prediction dataset (i.e no label for
	## observations)
	##
	## data_file is usualy the reduce formated scaled interpolated version
	## of the input data.
	##
	## map_matrix is the learned structure of the image.
	##
	## return the structure: {patient_id:[matrix]}
	## where matrix is a representation of the image associated to
	## the patient.
	##

	## importation
	import numpy
	import math

	## Init variables
	data_structure = {}

	## get cohorte data
	index_to_variable = {}
	cohorte = []
	input_data = open(data_file, "r")
	cmpt = 0
	for line in input_data:
		patient = {}
		patient_id = cmpt
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if(cmpt == 0):
			index = 0
			for variable in line_in_array:
				index_to_variable[index] = variable
				index += 1
		else:
			index = 0
			for scalar in line_in_array:
				patient[index_to_variable[index]] = int(scalar)
				index += 1
			cohorte.append(patient)
			data_structure[patient_id] = ["choucroute", "choucroute"]
		cmpt +=1
	input_data.close()

	## get map for the image structure
	variable_to_position = {}
	for x in range(0,len(map_matrix)):
		vector = map_matrix[x]
		for y in range(0,len(vector)):
			variable = map_matrix[x][y]
			position = [x,y]
			variable_to_position[variable] = position

	## create matrix for each patients in cohorte
	number_of_variables = len(index_to_variable.keys())
	side_size = math.sqrt(number_of_variables)
	side_size = int(side_size)
	cmpt = 0
	cohorte_matrix = []
	for patient in cohorte:

		## init patient grid
		patient_grid = numpy.zeros(shape=(side_size,side_size))
		patient_id = cmpt + 1

		## fill the patient grid
		for variable in index_to_variable.values():

			## get the variable id from information in the header
			variable_id = variable.split("_")
			variable_id = float(variable_id[-1])

			## get variable position in the grid
			position = variable_to_position[variable_id]
			position_x = position[0]
			position_y = position[1]

			## assign corresponding value to the position
			patient_grid[position_x][position_y] = int(patient[variable])

		## fill data structure
		data_structure[patient_id][0] = patient_grid
		cmpt +=1

	return data_structure
