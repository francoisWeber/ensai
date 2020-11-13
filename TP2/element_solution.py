def build_CNN(input_dim, output_dim, lr=0.001):
    #  Generate conv layers
    model = Sequential()

    # specify input dim :
    model.add(Input(shape=input_dim))

    # Add hidden layers

    # Conv2D : cf https://keras.io/api/layers/convolution_layers/convolution2d/
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu")),
    # MaxPool : cf https://keras.io/api/layers/pooling_layers/max_pooling2d/
    model.add(MaxPooling2D(pool_size=(2, 2))),
    # Dropout : cf https://keras.io/api/layers/regularization_layers/dropout/
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu")),
    model.add(MaxPooling2D(pool_size=(2, 2))),
    model.add(Dropout(rate=0.1))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu")),
    model.add(MaxPooling2D(pool_size=(2, 2))),
    model.add(Dropout(rate=0.1))

    # now flatten stuff to get a classical model
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(rate=0.3))

    # output layer
    model.add(Dense(output_dim, activation="softmax"))
    # /!\ Je ne sais pas pourquoi Romaric a mis une sortie linéaire plutôt qu'une softmax ..!!

    # pick and optimizer
    optimizer = RMSprop(learning_rate=lr)
    # Compile the network
    # https://keras.io/api/models/model_training_apis/#compile-method
    model.compile(
        loss="CategoricalCrossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model