import numpy as np
import datetime, os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def build_CNN(input_dim, output_dim, lr=0.001):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_dim))
	#Q1: Compléter ici
    model.add(Dense(output_dim))
	#Q1: Compilation du modèle
    return model


def build_CNN_from_pretrained(file_name, output_dim, lr=0.001):
    # Q4: compléter l'appel à la fonction load_model()
    model = load_model()
    # Q4: retirer et ajouter la/les couches nécessaires ici

    # Q4: Compilation du modèle
    return model


#Q2: Ajouter l'attribut rescale aux générateurs
train_generator = ImageDataGenerator()
valid_generator = ImageDataGenerator()

epochs = 15
batch_size = 64

#Q2: Ajouter les attributs "target_size" et "batch_size"
train_flow = train_generator.flow_from_directory("data/train", class_mode='categorical') 
valid_flow = train_generator.flow_from_directory("data/valid", class_mode='categorical')

from_scratch = True
if from_scratch:
    # Q2: compléter l'appel à la fonction build_CNN()
    cnn_model = build_CNN()
    label = 'from_scratch'
else:
    cnn_model = build_CNN_from_pretrained('pretrained_model.h5', 2)
    label = 'from_pretrained'
cnn_model.summary()

#Q2: compléter l'appel aux fonctions fit() et evaluate()
tensorboard_callback = TensorBoard(log_dir='logs/' + label, histogram_freq=int(epochs/10))
cnn_model.fit()
cnn_model.evaluate(verbose=2)
