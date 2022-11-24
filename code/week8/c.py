import time
import random

from matplotlib import pyplot as plt

from model import nnModel
from minheap import BtmkHeap


def populationBasedSearch(alpha, beta1, beta2, batchsize, epoches, iters, Nsample, bestP, exploitP, ii, tt, ff):
    # choose Nsample points
    global nnModel
    start_time = time.time()
    stp = BtmkHeap(bestP)
    i = 0
    z = 0
    while i < Nsample:
        calpha = random.uniform(alpha[0], alpha[1])
        cbeta1 = random.uniform(beta1[0], beta1[1])
        cbeta2 = random.uniform(beta2[0], beta2[1])
        cbatchsize = random.randint(batchsize[0], batchsize[1])
        cepoches = random.randint(epoches[0], epoches[1])
        print(str(i) + 'round: aplha=' + str(calpha) + ',beta1=' + str(cbeta1) + ',beta2=' + str(
            cbeta2) + ',batchsize=' + str(cbatchsize) + ',epoches=' + str(cepoches))
        model = nnModel(calpha, cbeta1, cbeta2, cbatchsize, cepoches)
        acc = model.getAuc()
        print(str(i) + 'round finished, get test data acc=' + str(acc))
        stp.Push((acc, [calpha, cbeta1, cbeta2, cbatchsize, cepoches]))
        i = i + 1

        used = (time.time() - start_time)
        # print(used)
        tt.append(used)
        ff.append(stp.getSmallest())

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

                minEpoches = paras[4] - 10
                if minEpoches < epoches[0]:
                    minEpoches = epoches[0]
                maxEpoches = paras[4] + 10
                if maxEpoches > epoches[1]:
                    maxEpoches = epoches[1]
                cepoches = random.randint(minEpoches, maxEpoches)
                print(str(i) + 'round,' + str(j) + 'exploit point: aplha=' + str(calpha) + ',beta1=' + str(
                    cbeta1) + ',beta2=' + str(
                    cbeta2) + ',batchsize=' + str(cbatchsize) + ',epoches=' + str(cepoches))
                model = nnModel(calpha, cbeta1, cbeta2, cbatchsize, cepoches)
                acc = model.getAuc()
                print(
                    str(i) + 'round finished,' + str(j) + 'exploit point:  get test data acc=' + str(stp.getSmallest()))
                stp.Push((acc, [calpha, cbeta1, cbeta2, cbatchsize, cepoches]))
                j = j + 1

        ii.append(i)
        used = (time.time() - start_time)
        # print(used)
        tt.append(used)
        ff.append(stp.getSmallest())
        i = i + 1

    paras = stp.getSmallestV()
    print('the best parameters are:aplha=' + str(paras[0]) + ',beta1=' + str(
        paras[1]) + ',beta2=' + str(
        paras[2]) + ',batchsize=' + str(paras[3]) + ',epoches=' + str(paras[4]))
    return ii, tt, ff


def globalRandomSearch(alpha, beta1, beta2, batchsize, epoches, iters, ii, tt, ff):
    start_time = time.time()
    minloss = float("inf")
    bestPara = (alpha[0], beta1[0], beta2[0], batchsize[0], epoches[0])
    i = 0
    while i < iters:
        calpha = random.uniform(alpha[0], alpha[1])
        cbeta1 = random.uniform(beta1[0], beta1[1])
        cbeta2 = random.uniform(beta2[0], beta2[1])
        cbatchsize = random.randint(batchsize[0], batchsize[1])
        cepoches = random.randint(epoches[0], epoches[1])
        print(str(i) + 'round: aplha=' + str(calpha) + ',beta1=' + str(cbeta1) + ',beta2=' + str(
            cbeta2) + ',batchsize=' + str(cbatchsize) + ',epoches=' + str(cepoches))
        model = nnModel(calpha, cbeta1, cbeta2, cbatchsize, cepoches)
        loss = model.getAuc()
        print(str(i) + 'round finished, get test data acc=' + str(minloss))

        if loss < minloss:
            minloss = loss
            bestPara = (calpha, cbeta1, cbeta2, cbatchsize, cepoches)

        ii.append(i)
        used = (time.time() - start_time)
        # print(used)
        tt.append(used)
        ff.append(minloss)
        i = i + 1

    print('best acc:' + str(minloss))
    print(bestPara)
    return ii, tt, ff


ii, tt, ff = globalRandomSearch([0, 0.3], [0, 1], [0, 1], [16, 128], [10, 50], 50, [], [], [])
ii2, tt2, ff2 = globalRandomSearch([0, 0.3], [0, 1], [0, 1], [16, 128], [10, 50], 120, [], [], [])
ii4, tt4, ff4 = populationBasedSearch([0, 1], [0, 1], [0, 1], [16, 128], [10, 50], 8, 10, 3, 3, [], [], [])
plt.figure(figsize=(7, 7))
plt.plot(tt, ff, color='b', label='global random search iters = 50')
plt.plot(tt2, ff2, color='r', label='global random search iters = 120')
plt.plot(ii4, ff4, color='b', label='population based search ')
plt.xlabel('iterations')
plt.ylabel('Changes in hyperparameters')
plt.legend()
plt.show()
