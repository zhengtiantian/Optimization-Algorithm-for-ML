
import random
from minheap import BtmkHeap
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
# from cifar10 import getmodel
# from MNIST import getmodel
from imbd import getmodel

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# n = 5000
# x_train = x_train[1:n];
# y_train = y_train[1:n]
# # x_test=x_test[1:500]; y_test=y_test[1:500]
#
# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255


# model = KerasClassifier(
# build_fn=getmodel
# )

# keras_param_options = {
#     'learning_rate': [0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
#     'beta_1': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
#     'beta_2': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
#     'batch_size':[16,32,64,128,256]
# }
#
# rs_keras = RandomizedSearchCV(
#     model,
#     param_distributions = keras_param_options,
#     scoring = 'neg_log_loss',
#     n_iter = 50,
#     cv = 3,
#     verbose = 1
# )
# rs_keras.fit(x_train, y_train,epochs=30)
#
# print('Best score obtained: {0}'.format(rs_keras.best_score_))
# print('Parameters:')
# for param, value in rs_keras.best_params_.items():
#     print('\t{}: {}'.format(param, value))


def populationBasedSearch(alpha, beta1, beta2, batchsize, iters, Nsample, bestP, exploitP):
    # choose Nsample points
    stp = BtmkHeap(bestP)
    i = 0
    while i < Nsample:
        calpha = random.uniform(alpha[0], alpha[1])
        cbeta1 = random.uniform(beta1[0], beta1[1])
        cbeta2 = random.uniform(beta2[0], beta2[1])
        cbatchsize = random.randint(batchsize[0], batchsize[1])
        print(str(i) + 'round: aplha=' + str(calpha) + ',beta1=' + str(cbeta1) + ',beta2=' + str(
            cbeta2) + ',batchsize=' + str(cbatchsize))
        # model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'cnn','adam')
        model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'cnn','constant')
        # model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'RL','adam')
        # model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'RL','constant')
        print(str(i) + 'round finished, get test data acc=' + str(loss))
        stp.Push((loss, [calpha, cbeta1, cbeta2, cbatchsize]))
        i = i + 1


    while i < iters + Nsample:
        datas = stp.BtmK()

        for data in datas:
            paras = data[1]
            j = 0
            while j < exploitP:
                minAlpha = paras[0] - 0.05
                if minAlpha < alpha[0]:
                    minAlpha = alpha[0]
                maxAlpha = paras[0] + 0.05
                if maxAlpha > alpha[1]:
                    maxAlpha = alpha[1]
                calpha = random.uniform(minAlpha, maxAlpha)

                minBeta1 = paras[1] - 0.1
                if minBeta1 < beta1[0]:
                    minBeta1 = beta1[0]
                maxBeta1 = paras[1] + 0.1
                if maxBeta1 > beta1[1]:
                    maxBeta1 = beta1[1]
                cbeta1 = random.uniform(minBeta1, maxBeta1)

                minBeta2 = paras[2] - 0.1
                if minBeta2 < beta2[0]:
                    minBeta2 = beta2[0]
                maxBeta2 = paras[2] + 0.1
                if maxBeta2 > beta1[1]:
                    maxBeta2 = beta1[1]
                cbeta2 = random.uniform(minBeta2, maxBeta2)

                minbatchsize = paras[3] - 5
                if minbatchsize < batchsize[0]:
                    minbatchsize = batchsize[0]
                maxbatchsize = paras[3] + 5
                if maxbatchsize > batchsize[1]:
                    maxbatchsize = batchsize[1]
                cbatchsize = random.randint(minbatchsize, maxbatchsize)


                print(str(i) + 'round,' + str(j) + 'exploit point: aplha=' + str(calpha) + ',beta1=' + str(
                    cbeta1) + ',beta2=' + str(
                    cbeta2) + ',batchsize=' + str(cbatchsize))
                # model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'cnn','adam')
                model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'cnn','constant')
                # model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize,30,'RL','adam')
                # model, history, loss = getmodel(calpha, cbeta1, cbeta2, cbatchsize, 30, 'RL', 'constant')
                print(
                    str(i) + 'round finished,' + str(j) + 'exploit point:  get test data acc=' + str(stp.getSmallest()))
                stp.Push((loss, [calpha, cbeta1, cbeta2, cbatchsize]))
                j = j + 1

        i = i + 1

    paras = stp.getSmallestV()
    print('the best parameters are:aplha=' + str(paras[0]) + ',beta1=' + str(
        paras[1]) + ',beta2=' + str(
        paras[2]) + ',batchsize=' + str(paras[3]))


populationBasedSearch([0, 1], [0, 1], [0, 1], [16, 128], 10, 10, 3, 3)
