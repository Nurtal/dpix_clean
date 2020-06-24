

def get_important_pixels_old(rgb_array):
    """

    >DEPRECATED AFTER AN UPDATE OF KERAS-VIS<

    Read a 3d numpy array containing a list of rgb values.

    Return a list of tuple where the first element is the position of the pixel
    in the grid in the form: y_x and the second element the associated score,
    compute within the function.

    The more "hot" (red, orange, yellow ...) is the pixel, the highest is the
    computed score.

    -> exemple structure ror rgb_array : [[[255,125,0],[221,246,35]],
                                          [[255,255,255],[125,255,32]],
                                          [[10,0,0],[75,0,0]]]

    WARNINGS : The score computation is custom, probably not perfect, for now it
    identify intersting pixels but have trouble to rank them with precisions.
    """

    ## importation
    from matplotlib import pyplot as plt
    import numpy as np
    import math
    import operator

    ## parameters
    rgb_array = np.asarray(rgb_array)
    pos_to_score = {}

    ## find the top red pixels
    ## TODO: the current score computation used (option 2)
    ## spot also intense yellow pixel, might have to tune this
    ## stuff
    y = 0
    for vector in rgb_array:
        x = 0
        for pixel in vector:


            ## compute score
            red_compo = pixel[0]
            green_compo = pixel[1]
            blue_compo = pixel[2]
            alpha_coef = 2
            delta = (abs(blue_compo-127) + abs(red_compo - green_compo))
            score = red_compo*alpha_coef * delta

            ## assign value
            pos_to_score[str(y)+"_"+str(x)] = score
            x +=1
        y+=1

    ## sort dictionnary and return ordered list of tuples (position, score)
    sorted_values = sorted(pos_to_score.items(), key=operator.itemgetter(1))
    sorted_values.reverse()
    return sorted_values



def get_important_pixels(heatmap_array):
    """
    Read a 2d numpy array containing a list of pixel importance.

    Return a list of tuple where the first element is the position of the pixel
    in the grid in the form: y_x and the second element the associated score,
    retrieved within the function.
    """

    ## importation
    from matplotlib import pyplot as plt
    import numpy as np
    import math
    import operator

    ## parameters
    heatmap_array = np.asarray(heatmap_array)
    pos_to_score = {}

    ## assign a score to each position
    y = 0
    for vector in heatmap_array:
        x = 0
        for score in vector:
            ## assign value
            pos_to_score[str(y)+"_"+str(x)] = score
            x +=1
        y+=1

    ## sort dictionnary and return ordered list of tuples (position, score)
    sorted_values = sorted(pos_to_score.items(), key=operator.itemgetter(1))
    sorted_values.reverse()
    return sorted_values



def save_features_importance(grid_matrix, pos_to_score, modifier, class_id, input_type):
    """
    Create a log file ( in the log folder ) and write the following information
    for each feature :
        -> feature id (used in the grid matrix)
        -> feature name (extract from the variable manifest)
        -> score (compute from the get_important_pixels function)

    - grid_matrix is a 2d numpy array : the image structure
    - pos_to_score is a list of tuple, get from get_important_pixels function
    - modifier is a string, the backprop modifier used to compute saliency
    - class_id can be anything, represent the label of the class (usually an int)
    - input_type is a string, the origin of the input : can be
                -> saliency
                -> gradCam
    """

    ## parameters
    feature_name = "NA"
    log_file_name = "log/"+str(class_id)+"_feature_importance_"+str(input_type)+"_"+str(modifier)+".log"

    ## open log file and write header
    log_file = open(log_file_name, "w")
    log_file.write("id,name,score\n")

    ## Fill the log file
    for info in pos_to_score:

        ## get score
        score = info[1]

        ## get position
        position = info[0]
        position = position.split("_")
        y = int(position[0])
        x = int(position[1])

        ## get feature id
        feature_id = int(grid_matrix[y][x])

        ## get feature name
        manifest = open("variable_manifest.csv", "r")
        for line in manifest:
            line = line.rstrip()
            line_in_array = line.split(",")
            id_to_test = line_in_array[1].replace("variable_", "")
            if(int(id_to_test) == feature_id):
                feature_name = line_in_array[0]
        manifest.close()
        log_file.write(str(feature_id)+","+str(feature_name)+","+str(score)+"\n")

    ## close log file
    log_file.close()
