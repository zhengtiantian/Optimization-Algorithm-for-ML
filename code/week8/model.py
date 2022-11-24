from keras import regularizers
from keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow import keras


class nnModel(object):
    def __init__(self, alpha, beta1, beta2, batchsize, epoches):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.batchsize = batchsize
        self.epoches = epoches

        self.num_classes = 10
        self.input_shape = (32, 32, 3)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        n = 5000
        x_train = x_train[1:n];
        y_train = y_train[1:n]
        # x_test=x_test[1:500]; y_test=y_test[1:500]

        # Scale images to the [0, 1] range
        self.x_train = x_train.astype("float32") / 255
        self.x_test = x_test.astype("float32") / 255
        # print("orig x_train shape:", x_train.shape)

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

    def getAuc(self):
        model = keras.Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))

        adam = keras.optimizers.Adam(learning_rate=self.alpha, beta_1=self.beta1, beta_2=self.beta2)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])


        batch_size = self.batchsize
        epochs = self.epoches
        history = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                            verbose=0)

        loss = model.evaluate(self.x_test, self.y_test)[0]
        if loss > 5:
            loss = 5
        return loss
