
import random
from minheap import BtmkHeap
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from cifar10 import getmodel



model = KerasClassifier(
build_fn=getmodel
)

keras_param_options = {
    'learning_rate': [0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    'beta_1': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
    'beta_2': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999],
    'batch_size':[16,32,64,128,256]
}

rs_keras = RandomizedSearchCV(
    model,
    param_distributions = keras_param_options,
    scoring = 'neg_log_loss',
    n_iter = 50,
    cv = 3,
    verbose = 1
)
# rs_keras.fit(x_train, y_train,epochs=30)

print('Best score obtained: {0}'.format(rs_keras.best_score_))
print('Parameters:')
for param, value in rs_keras.best_params_.items():
    print('\t{}: {}'.format(param, value))



