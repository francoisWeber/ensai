import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical


## One hot encode the columns of the database specified in list_id.
#
#  return a copy of the database, on hot encoded. 
def one_hot_encode_database(database, list_id):
    encoded_database = np.empty(shape=(database.shape[0], 0), dtype=float)
    for id in range(database.shape[1]):
        if id in list_id:
            original_column = database[:, id]
            encoded_column = to_categorical(original_column.astype(int), num_classes=int(np.max(original_column)+1))
            encoded_database = np.column_stack((encoded_database, encoded_column))
        else:
            original_column = database[:, id]
            encoded_database = np.column_stack((encoded_database, original_column))
    return encoded_database

## Normalize between 0 and 1 each column of the database specified in list_id.
#
#  return a copy of the database, normalized. 
def normalize_database(database, list_id):
    encoded_database = database.copy()
    for id in list_id:
        encoded_database[:, id] = (encoded_database[:, id] - np.amin(encoded_database[:, id])) / (np.amax(encoded_database[:, id]) - np.amin(encoded_database[:, id]))
    return encoded_database

## Split the input data into nb_fold equal folds. Select current_fold as the test_dataset and the stack the other folds to be the train_dataset.
# 
#  return a training_dataset and a test_dataset.
def get_kfold_cv(input_data, nb_fold, current_fold):
    folds = np.array_split(input_data, nb_fold, axis=0)
    test_dataset = folds[current_fold]
    training_dataset = np.empty((0, test_dataset.shape[1]), dtype=float)
    for i in range(0,nb_fold):
        if i != current_fold:
            training_dataset = np.append(training_dataset, folds[i], axis=0)
    return training_dataset, test_dataset

## Construct a layer composed of dense layers, which dimensions are definded in the layer_list argument.
# 
#  return the constructed and compiled model.
def build_NN(layer_list, input_dim, output_dim, lr=0.001):
    # Q7: add layers and "compile" the model
    model = Sequential()
    #Add layers
    model.add()
	
	#Compile the network
    model.compile(loss=None, optimizer=None, metrics=None)
    return model
