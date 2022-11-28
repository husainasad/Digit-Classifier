from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from scipy.ndimage.interpolation import shift
from tensorflow.keras.models import load_model
import numpy as np


def getDataset():
 	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
 	Xtrain = Xtrain.reshape((Xtrain.shape[0], 28, 28, 1))
 	Xtest = Xtest.reshape((Xtest.shape[0], 28, 28, 1))
 	Ytest = to_categorical(Ytest)
 	Ytrain = to_categorical(Ytrain)
 	return Xtrain, Ytrain, Xtest, Ytest


def processData(train, test):
 	train = train.astype('float32')
 	test = test.astype('float32')
 	train = train / 255.0
 	test = test / 255.0
 	return train, test


def constructModel():
 	model = Sequential()
 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 	model.add(MaxPooling2D((2, 2)))
 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
 	model.add(MaxPooling2D((2, 2)))
 	model.add(Flatten())
 	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
 	model.add(Dense(10, activation='softmax'))
 	opt = SGD(learning_rate=0.01, momentum=0.9)
 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 	return model


def modelEvaluation(dataX, dataY, n_folds=5):
 	scores, histories = list(), list()
 	kfold = KFold(n_folds, shuffle=True, random_state=1)
 	for train_ix, test_ix in kfold.split(dataX):
          model = constructModel()
          trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
          history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
          _, acc = model.evaluate(testX, testY, verbose=0)
          print('> %.3f' % (acc * 100.0))
          scores.append(acc)
          histories.append(history)
 	return scores, histories


def getDiagnostics(histories):
 	for i in range(len(histories)):
         plt.subplot(2, 1, 1)
         plt.title('Cross Entropy Loss')
         plt.plot(histories[i].history['loss'], color='blue', label='train')
         plt.plot(histories[i].history['val_loss'], color='orange', label='test')
         plt.subplot(2, 1, 2)
         plt.title('Classification Accuracy')
         plt.plot(histories[i].history['acc'], color='blue', label='train')
         plt.plot(histories[i].history['val_acc'], color='orange', label='test')
 	plt.show()


def getPerformance(scores):
 	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
 	plt.boxplot(scores)
 	plt.show()


def startTraining():
    trainX, trainY, testX, testY = getDataset()
    trainX, testX = processData(trainX, testX)
    scores, histories = modelEvaluation(trainX, trainY)
    getDiagnostics(histories)
    getPerformance(scores)
    model = constructModel()
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    model.save('DigitClassifier.h5')

startTraining()

def startTesting():
 	trainX, trainY, testX, testY = getDataset()
 	trainX, testX = processData(trainX, testX)
 	model = load_model('DigitClassifier.h5')
 	_, acc = model.evaluate(testX, testY, verbose=0)
 	print('> %.3f' % (acc * 100.0))

startTesting()