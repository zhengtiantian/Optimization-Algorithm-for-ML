import numpy as np
import tensorflow as tf
from keras import backend
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

plt.rcParams['figure.constrained_layout.use'] = True
import sys

# Model / data parameters
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
n = 50000
x_train = x_train[1:n];
y_train = y_train[1:n]
# x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

# x_train = x_train.reshape((-1, 784))
# x_test = x_test.reshape((-1, 784))

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def getmodel(learning_rate, beta_1, beta_2, batch_size, epochs, modelName, opti):
    backend.clear_session()
    model = keras.Sequential()
    if modelName == 'cnn':
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), input_shape=x_train.shape[1:], activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', activity_regularizer=regularizers.l2(0.01)))
    else:
        model.add(Dense(num_classes, activation='softmax', input_dim=x_train.shape[1],
                        activity_regularizer=regularizers.l2(0.01)))

    optimizer = ''
    if opti == 'adam':
        # optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.1, beta_2=0.1)
        optimizer = keras.optimizers.Adam()
    else:
        optimizer = keras.optimizers.SGD(momentum=0.0, nesterov=False)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    print(model.evaluate(x_test, y_test))
    print(model.evaluate(x_train, y_train))
    loss = model.evaluate(x_test, y_test)[0]
    return model, history, loss


# model, history, loss = getmodel(0, 0, 0, 128, 30, 'cnn', 'adam')
# model1, history1, loss1 = getmodel(0, 0, 0, 128, 30, 'cnn', 'constant')
# model, history, loss = getmodel(0, 0, 0, 128, 30, 'RL', 'adam')
# model1, history1, loss1 = getmodel(0, 0, 0, 128, 30, 'RL', 'constant')
#
#
# plt.subplot(211)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.plot(history1.history['accuracy'])
# plt.plot(history1.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train adam', 'val adam', 'train constant', 'val constant'])
# plt.subplot(212)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss');
# plt.xlabel('epoch')
# plt.legend(['train adam', 'val adam', 'train constant', 'val constant'])
# plt.show()

model1, history1, loss1 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'constant')
model2, history2, loss2 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'constant')
model3, history3, loss3 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'constant')
model4, history4, loss4 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'adam')
model5, history5, loss5 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'adam')
model6, history6, loss6 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'adam')
# model1, history1, loss1 = getmodel(0, 0, 0.0, 128, 30, 'RL', 'constant')
# model2, history2, loss2 = getmodel(0, 0, 0.0, 128, 30, 'RL', 'constant')
# model3, history3, loss3 = getmodel(0, 0, 0.0, 128, 30, 'RL', 'constant')
# model4, history4, loss4 = getmodel(0, 0, 0.0, 128, 30, 'RL', 'adam')
# model5, history5, loss5 = getmodel(0, 0, 0.0, 128, 30, 'RL', 'adam')
# model6, history6, loss6 = getmodel(0, 0, 0.0, 128, 30, 'RL', 'adam')
plt.plot(history1.history['val_loss'])
plt.plot(history2.history['val_loss'])
plt.plot(history3.history['val_loss'])
plt.plot(history4.history['val_loss'])
plt.plot(history5.history['val_loss'])
plt.plot(history6.history['val_loss'])
plt.title('model loss')
plt.ylabel('val loss');
plt.xlabel('epoch')
plt.legend(['val loss constant1', 'val loss constant2', 'val loss constant3', 'val loss adam1', 'val loss adam2',
            'val loss adam3'])
plt.show()
