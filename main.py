import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.preprocessing.text import Tokenizer
import hashlib
import copy

from keras.optimizers.legacy import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score


def categorize_to_classes(input, output):
    classes = np.unique(output)
    c1 = input[output == classes[0], :]
    c2 = input[output == classes[1], :]
    c3 = input[output == classes[2], :]
    c4 = input[output == classes[3], :]
    c5 = input[output == classes[4], :]

    plt.figure()
    plt.plot(c1[:, 0], c1[:, 1], 'o')
    plt.plot(c2[:, 0], c2[:, 1], 'o')
    plt.plot(c3[:, 0], c3[:, 1], 'o')
    plt.plot(c4[:, 0], c4[:, 1], 'o')
    plt.plot(c5[:, 0], c5[:, 1], 'o')

    plt.title('Data divided into classes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['politics', 'sports', 'tech', 'entertainment', 'business'])
    plt.grid()
    plt.show()

    plt.figure()
    plt.hist(output)
    plt.title('Histogram representing input data count as per classes')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    return classes


def tokenize_to_matrix(train_text_data, test_train_data):
    t = Tokenizer()
    t.fit_on_texts(train_text_data)  # Vocabulary is built on training data

    # It is more useful to have word frequencies rather than word counts
    return t.texts_to_matrix(train_text_data, mode='tfidf'), t.texts_to_matrix(test_train_data, mode='tfidf')


def scale_min_max(encoded_training_transform, encoded_test_transform):
    scaler = MinMaxScaler()
    scaler.fit(encoded_training_transform)

    return scaler.transform(encoded_training_transform), scaler.transform(encoded_test_transform)


# def find_best_params(encoded_input_training_scaled, output_training_oh):
#     model = KerasClassifier(model=make_model(encoded_input_training_scaled.shape[1], output_training_oh.shape[1]),
#                             verbose=0)
#     params = {
#         'model__learning_rate': [0.001, 0.01, 0.1],
#         'model__activation': ['relu', 'tanh', 'sigmoid'],
#         'batch_size': [8, 16, 32],
#         'epochs': [50, 100, 1000]
#     }
#     gs = GridSearchCV(estimator=model, param_grid=params, verbose=5, cv=3, n_jobs=-1)
#     gs = gs.fit(encoded_input_training_scaled, output_training_oh)
#
#     print('Optimal activation function: ', gs.best_params['model__activation'])
#     print('Optimal learning rate: ', gs.best_params['model__learning_rate'])
#     print('Optimal batch_size: ', gs.best_params['batch_size'])
#     print('Optimal epoch_number: ', gs.best_params['epochs'])
#
#     # AttributeError: 'Adam' object has no attribute 'build'
#
#     return gs


def make_model(n_in, n_out, learning_rate=0.1, activation='relu'):
    model = Sequential()
    model.add(Dense(10, input_dim=n_in, activation=activation))
    model.add(Dense(5, activation=activation))
    model.add(Dense(n_out, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_network(input, output, classes):
    output_oh = to_categorical(output)
    input_text = input.iloc[:, 0].to_numpy()

    # divide into training and test sets
    input_training, input_test, output_training_oh, output_test_oh = (
        train_test_split(input_text, output_oh, shuffle=True, test_size=0.2, random_state=47))

    # Since we're working with text as input it is necessary to encode it using Tokenizer API from keras
    encoded_input_training, encoded_input_test = tokenize_to_matrix(input_training, input_test)

    # Now it is necessary to scale frequencies to 0-1 aspect

    encoded_input_training_scaled, encoded_input_test_scaled = scale_min_max(encoded_input_training, encoded_input_test)

    # Early stopping
    early_stopping = (EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True))

    # gs = find_best_params(encoded_input_training_scaled, output_training_oh)

    model = make_model(encoded_input_training_scaled.shape[1], output_training_oh.shape[1])

    history = model.fit(encoded_input_training_scaled, output_training_oh, epochs=1000,
                        batch_size=32,
                        validation_data=(encoded_input_test_scaled, output_test_oh), callbacks=[early_stopping],
                        verbose=0)

    print('Model precision: ' + str(
        100 * model.evaluate(encoded_input_test_scaled, output_test_oh, verbose=1)[1]) + '%.')

    # trening/validaciona kriva
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Function loss over epochs')
    plt.ylabel('Function loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])
    plt.show()

    # konf matrica

    y_true = np.argmax(output_test_oh, axis=1)
    y_pred = np.argmax(model.predict(encoded_input_test_scaled), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['politics', 'sports', 'tech', 'entertainment', 'business']).plot()
    plt.show()

    # osetljivost/preciznost

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    print("Precision by classes:" + str(precision))
    print("Sensitivity by classes:" + str(recall))
    print("Precision accuracy: " + str(avg_precision))
    print("Average sensitivity: " + str(avg_recall))

    # granica odlucivanja
    Ntest = 100
    x = np.linspace(-10, 7, Ntest)
    y = np.linspace(-10, 7, Ntest)
    Xgrid, Ygrid = np.meshgrid(x, y)
    grid = np.c_[Xgrid.ravel(), Ygrid.ravel()]
    _, grid_skaliran = scale_min_max(encoded_input_training, grid)
    ypred = model.predict(grid_skaliran, verbose=0)
    ypred = np.argmax(ypred, axis=1)
    c0pred = grid[ypred == 0, :]
    c1pred = grid[ypred == 1, :]
    c2pred = grid[ypred == 2, :]
    c3pred = grid[ypred == 3, :]
    c4pred = grid[ypred == 4, :]

    plt.figure()
    plt.plot(c0pred[:, 0], c0pred[:, 1], 'p.', alpha=0.1)
    plt.plot(c1pred[:, 0], c1pred[:, 1], 's.', alpha=0.1)
    plt.plot(c2pred[:, 0], c2pred[:, 1], 't.', alpha=0.1)
    plt.plot(c3pred[:, 0], c2pred[:, 1], 'e.', alpha=0.1)
    plt.plot(c4pred[:, 0], c2pred[:, 1], 'b.', alpha=0.1)
    plt.plot(classes[0][:, 0], classes[0][:, 1], 'po')
    plt.plot(classes[1][:, 0], classes[1][:, 1], 'so')
    plt.plot(classes[2][:, 0], classes[2][:, 1], 'to')
    plt.plot(classes[3][:, 0], classes[3][:, 1], 'eo')
    plt.plot(classes[4][:, 0], classes[4][:, 1], 'bo')
    plt.title('Predicted decision boundaries and original data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-4, 7])
    plt.ylim([-4, 7])
    plt.show()


def main():
    data = pd.read_csv('df_file.csv', header=None)

    data.columns = ['x', 'y']
    data = data.iloc[1:]  # just ignore first row

    input = data.iloc[:, :2]

    input_deep_copy = copy.deepcopy(input)

    input_text = input.iloc[:, 0].to_numpy()

    for i in range(len(input_text)):
        input_text[i] = int(hashlib.sha256(input_text[i].encode('utf-8')).hexdigest(), 16) % 10 ** 256

    # hash text inputs so they can be represented on a graph

    input = input.to_numpy()
    output = data.y.to_numpy()

    classes = categorize_to_classes(input, output)
    train_network(input_deep_copy, output, classes)


if __name__ == "__main__":
    main()
