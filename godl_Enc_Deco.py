# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:08:17 2019

@author: Sohail Khan
"""

import pandas as pd 
import numpy as np
from imutils import paths
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.preprocessing import AspectAwarePreprocessor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import L1L2,l1_l2,l1,l2
from keras import backend as K
from spp.SpatialPyramidPooling import SpatialPyramidPooling
import argparse
import os
import shutil
#from keras_utils import reset_tf_session
import keras.utils

#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import confusion_matrix

def build_deep_autoencoder(img_shape, code_size):
    """PCA's deeper brother. See instructions above. Use `code_size` in layer definitions."""
    H,W,C = img_shape
    #print('img_shape',img_shape)
    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Conv2D(32, kernel_size=(3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    encoder.add(L.Conv2D(64, kernel_size=(3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    encoder.add(L.Conv2D(128, kernel_size=(3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    encoder.add(L.Conv2D(256, kernel_size=(3,3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))
    ### YOUR CODE HERE: define encoder as per instructions above ###

    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(2*2*256))
    decoder.add(L.Reshape((2,2,256)))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2,  padding='same'))
    ### YOUR CODE HERE: define decoder as per instructions above ###
    
    #encoder.summary()
    #decoder.summary()
    return encoder, decoder

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        #first CONV=>RELU=>CONV=>RELU=>POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape, kernel_regularizer=l2(0.0001)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))        
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))        
        model.add(Dropout(0.25))
        
        # second CONV=>RELU=>CONV=>RELU=>POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(SpatialPyramidPooling([1, 2, 4, 6, 8]))
        # first (and only) set of FC => RELU layers
#        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        model.summary()
        # return the constructed network architecture
        return model


trainDataFrame = pd.read_csv( "./train/train.csv", delimiter=',')
testDataFrame = pd.read_csv( "./test/test.csv")


labelNames = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tankers']
#for l in labelNames:
#    dirName = './train/'+l
#    print(dirName)
#    if not os.path.exists(dirName):
#        os.makedirs(dirName)
#
#for tdf in range(trainDataFrame.shape[0]):
#    file = trainDataFrame.iloc[tdf,0]
#    if trainDataFrame.iloc[tdf,1] == 1:
#        print(file)
#        shutil.copy(folderPath + 'train/images/'+file,folderPath + 'train/Cargo/')
#        
#    if trainDataFrame.iloc[tdf,1] == 2:
#        print(file)
#        shutil.copy(folderPath + 'train/images/'+file,folderPath + 'train/Military/')
#        
#    if trainDataFrame.iloc[tdf,1] == 3:
#        print(file)
#        shutil.copy(folderPath + 'train/images/'+file,folderPath + 'train/Carrier/')
#        
#    if trainDataFrame.iloc[tdf,1] == 4:
#        print(file)
#        shutil.copy(folderPath + 'train/images/'+file,folderPath + 'train/Cruise/')
#        
#    if trainDataFrame.iloc[tdf,1] == 5:
#        print(file)
#        shutil.copy(folderPath + 'train/images/'+file,folderPath + 'train/Tankers/')
        
        
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default=  "./train/test", help="path to the input dataset")
ap.add_argument("-o", "--output", required=False, default= "./train/gameofdeeplearning.png", help="path to save the plot")
ap.add_argument("-m", "--model", required=False, default=  "./train/gameofdeeplearning.hdf5", help="path to save the model")
args = vars(ap.parse_args())


#grab the list of images that we'll be describing
print("[INFO] laoding images...")
imagePaths = list(paths.list_images(args["dataset"]))



# initialize the image preprocessors
aap = AspectAwarePreprocessor(210, 210)
iap = ImageToArrayPreprocessor()
#sp = SimplePreprocessor(200, 145)
#iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixels intensities to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 70% of
# the data for training and the remaining 30% for testing
(trainX, testX, trainY, testY) = train_test_split(data, data, test_size=0.30, random_state=42)

## convert the labels from integers to vectors
#trainY = LabelBinarizer().fit_transform(trainY)
#testY = LabelBinarizer().fit_transform(testY)

IMG_SHAPE = (210, 210, 3)
encoder, decoder = build_deep_autoencoder(IMG_SHAPE, code_size=32)

inp = L.Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = keras.models.Model(inputs=inp, outputs=reconstruction)
autoencoder.summary()
autoencoder.compile(optimizer='adamax', loss='mse')

autoencoder.fit(x=trainX, y=trainX, epochs=15,
                validation_data=[testX, testX],
                verbose=0)
#autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))













## initialize the optimizer and model
#print("[INFO] compiling model...")
#model = MiniVGGNet.build(width=200, height=150, depth=3, classes=len(labelNames))
#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=25, min_lr=0.000001, verbose=1)
#checkpointer = ModelCheckpoint(filepath="./train/checkpoint.hdf5", verbose=1, save_best_only=True, mode='max')
#sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
#
## train the network
#print("[INFO] training network...")
#H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, 
#              verbose=1,callbacks=[reduce_lr, checkpointer], shuffle=True)
#
## save the network to disk
#print("[INFO] serializing network...")
#model.save(args["model"])
#
#
## evaluate the network
#print("[INFO] evaluating network...")
#predictions = model.predict(testX, batch_size=32)
#print(classification_report(testY.argmax(axis=1),
#predictions.argmax(axis=1), target_names=labelNames))
#
## plot the training loss and accuracy
#plt.style.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
#plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend()
#plt.savefig(args["output"], dpi=600)
#plt.show()
#
#




## predict probabilities for test set
#yhat_probs = model.predict(testX, verbose=0)
## predict crisp classes for test set
#yhat_classes = model.predict_classes(testX, verbose=0)
## reduce to 1d array
#yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]
# 
## accuracy: (tp + tn) / (p + n)
#accuracy = accuracy_score(testy, yhat_classes)
#print('Accuracy: %f' % accuracy)
## precision tp / (tp + fp)
#precision = precision_score(testy, yhat_classes)
#print('Precision: %f' % precision)
## recall: tp / (tp + fn)
#recall = recall_score(testy, yhat_classes)
#print('Recall: %f' % recall)
## f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(testy, yhat_classes)
#print('F1 score: %f' % f1)
# 
## kappa
#kappa = cohen_kappa_score(testy, yhat_classes)
#print('Cohens kappa: %f' % kappa)
## ROC AUC
#auc = roc_auc_score(testy, yhat_probs)
#print('ROC AUC: %f' % auc)
## confusion matrix
#matrix = confusion_matrix(testy, yhat_classes)
#print(matrix)