from keras import backend
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
import matplotlib.pyplot as plt

# plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
n = 50000
x_train = x_train[1:n];
y_train = y_train[1:n]
# x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("orig x_train shape:", x_train.shape)

# RL data
# x_train = x_train.reshape((-1, 3072))
# x_test = x_test.reshape((-1, 3072))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def getmodel(learning_rate, beta_1, beta_2, batch_size, epochs, modelName, opti):
    backend.clear_session()
    model = keras.Sequential()

    if modelName == 'cnn':
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    else:
        model.add(Dense(num_classes, activation='softmax', input_dim=x_train.shape[1]))

    optimizer = ''
    if opti == 'adam':
        # optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        optimizer = keras.optimizers.Adam()
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    loss = model.evaluate(x_test, y_test)[0]
    return model, history, loss


model, history, loss = getmodel(0.05, 0, 0.0, 128, 50, 'cnn', 'constant')
model1, history1, loss1 = getmodel(0, 0, 0.0, 128, 50, 'cnn', 'adam')
# model, history, loss = getmodel(0.05, 0, 0.0, 128, 25, 'RL', 'constant')
# model1, history1, loss1 = getmodel(0, 0, 0.0, 128, 25, 'RL', 'adam')
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train constant', 'val constant', 'train adam', 'val adam'])
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss');
plt.xlabel('epoch')
plt.legend(['train constant', 'val constant', 'train adam', 'val adam'])
plt.show()

# model1, history1, loss1 = getmodel(0.05, 0, 0.0, 128, 25, 'cnn', 'constant')
# model2, history2, loss2 = getmodel(0.05, 0, 0.0, 128, 25, 'cnn', 'constant')
# model3, history3, loss3 = getmodel(0.05, 0, 0.0, 128, 25, 'cnn', 'constant')
# model4, history4, loss4 = getmodel(0, 0, 0.0, 128, 25, 'cnn', 'adam')
# model5, history5, loss5 = getmodel(0, 0, 0.0, 128, 25, 'cnn', 'adam')
# model6, history6, loss6 = getmodel(0, 0, 0.0, 128, 25, 'cnn', 'adam')
# model1, history1, loss1 = getmodel(0.05, 0, 0.0, 128, 25, 'RL', 'constant')
# model2, history2, loss2 = getmodel(0.05, 0, 0.0, 128, 25, 'RL', 'constant')
# model3, history3, loss3 = getmodel(0.05, 0, 0.0, 128, 25, 'RL', 'constant')
# model4, history4, loss4 = getmodel(0, 0, 0.0, 128, 25, 'RL', 'adam')
# model5, history5, loss5 = getmodel(0, 0, 0.0, 128, 25, 'RL', 'adam')
# model6, history6, loss6 = getmodel(0, 0, 0.0, 128, 25, 'RL', 'adam')
# plt.plot(history1.history['val_loss'])
# plt.plot(history2.history['val_loss'])
# plt.plot(history3.history['val_loss'])
# plt.plot(history4.history['val_loss'])
# plt.plot(history5.history['val_loss'])
# plt.plot(history6.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('val loss');
# plt.xlabel('epoch')
# plt.legend(['val loss constant1', 'val loss constant2', 'val loss constant3', 'val loss adam1', 'val loss adam2',
#             'val loss adam3'])
# plt.show()
# results = model.evaluate(x_test, y_test)
# print(results)
