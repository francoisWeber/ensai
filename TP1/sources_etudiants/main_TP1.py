from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from fonctions_TP1 import *


database = np.genfromtxt("dataset/wine_database.csv", delimiter=",", dtype=float)

# Q7: one hot encode the class and normalize the features of the database here

np.random.shuffle(database)

nb_fold = 5
layer_list = [50, 50, 20]

quick_check = True

if quick_check:
    # Q10: quick check
    model = build_NN(layer_list, database.shape[1] - 3, 3, lr=0.1)
    model.fit(database[:, 3:], database[:, :3], batch_size=16, epochs=100)
    model.evaluate(database[:, 3:], database[:, :3])
else:
    gt_and_pred = np.empty(shape=(2, 0))

    for fold in range(nb_fold):
        # Q11: test several learning rates
        model = build_NN(layer_list, database.shape[1]-3, 3, lr=0.1)
        model.summary()
        training_fold, valid_fold = get_kfold_cv(database, nb_fold, fold)
        callback = EarlyStopping(patience=5)
        # Q13: add EarlyStopping() callback
        model.fit(training_fold[:, 3:], training_fold[:, :3], batch_size=16, epochs=200, validation_data=(valid_fold[:, 3:], valid_fold[:, :3]), shuffle=True)
        fold_prediction = model.predict(valid_fold[:, 3:])
        gt_and_pred = np.column_stack((gt_and_pred, np.array([np.argmax(valid_fold[:, :3], axis=1), np.argmax(fold_prediction, axis=1)])))

    # Q12: display confusion matrix and balanced accuracy here
