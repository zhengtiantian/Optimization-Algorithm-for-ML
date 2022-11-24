import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


plt.rcParams['figure.constrained_layout.use'] = True
import sys

word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])



# the data, split between train and test sets
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)


#Code copied from function sortTableArray obtained from https://tensorflow.google.cn/tutorials/keras/text_classification
print("orig x_train shape:", train_data.shape)
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# x_test=x_test[1:500]; y_test=y_test[1:500]


print("orig x_train shape:", train_data.shape)


def getmodel(learning_rate, beta_1, beta_2, batch_size, epochs, modelName, opti):

    model = keras.Sequential()
    # Code copied from function sortTableArray obtained from https://tensorflow.google.cn/tutorials/keras/text_classification
    vocab_size = 10000
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())

    if modelName == 'cnn':
        # Code copied from function sortTableArray obtained from https://tensorflow.google.cn/tutorials/keras/text_classification
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(keras.layers.Dense(1, activation='sigmoid',activity_regularizer=regularizers.l1(0.0001)))

    else:
        model.add(Dense(1, activation='sigmoid', input_dim=train_data.shape[1], activity_regularizer=regularizers.l1(0.0001)))

    optimizer = ''
    if opti == 'adam':
        # optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        optimizer = keras.optimizers.Adam()
    else:
        # optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0, nesterov=False)
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.0, nesterov=False)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()
    history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    print(model.evaluate(test_data, test_labels))
    print(model.evaluate(train_data, train_labels))
    loss = model.evaluate(test_data, test_labels)[0]
    return model, history, loss

# model, history, loss = getmodel(0.1, 0, 0, 128, 50, 'cnn', 'constant')
# model1, history1, loss1 = getmodel(0.0, 0, 0, 128, 50, 'cnn', 'adam')
# model, history, loss = getmodel(0.3, 0, 0, 128, 50, 'RL', 'constant')
# model1, history1, loss1 = getmodel(0.0, 0, 0, 128, 50, 'RL', 'adam')
# plt.subplot(211)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.plot(history1.history['accuracy'])
# plt.plot(history1.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train constant', 'val constant','train adam', 'val adam'])
# plt.subplot(212)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss');
# plt.xlabel('epoch')
# plt.legend(['train constant', 'val constant','train adam', 'val adam'])
# plt.show()


model1, history1, loss1 = getmodel(0.1, 0, 0.0, 128, 30, 'cnn', 'constant')
model2, history2, loss2 = getmodel(0.1, 0, 0.0, 128, 30, 'cnn', 'constant')
model3, history3, loss3 = getmodel(0.1, 0, 0.0, 128, 30, 'cnn', 'constant')
model4, history4, loss4 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'adam')
model5, history5, loss5 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'adam')
model6, history6, loss6 = getmodel(0, 0, 0.0, 128, 30, 'cnn', 'adam')
# model1, history1, loss1 = getmodel(0.3, 0, 0.0, 128, 50, 'RL', 'constant')
# model2, history2, loss2 = getmodel(0.3, 0, 0.0, 128, 50, 'RL', 'constant')
# model3, history3, loss3 = getmodel(0.3, 0, 0.0, 128, 50, 'RL', 'constant')
# model4, history4, loss4 = getmodel(0, 0, 0.0, 128, 50, 'RL', 'adam')
# model5, history5, loss5 = getmodel(0, 0, 0.0, 128, 50, 'RL', 'adam')
# model6, history6, loss6 = getmodel(0, 0, 0.0, 128, 50, 'RL', 'adam')
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

