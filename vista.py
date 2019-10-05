import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)
from keras.models import Sequential,load_model,Model
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import os
import cv2
import pickle
from collections import defaultdict
IMG_SIZE = 100
X,Y = np.load("trainX.npy"),np.load("trainY.npy")
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=7)

# Normalize image vectors
X_train = X_train / 255.
X_test = X_test / 255.
# Convert training and test labels to one hot matrices
# 6. Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 14)
Y_test = np_utils.to_categorical(Y_test, 14)
print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
from keras import optimizers
from keras import applications
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (IMG_SIZE, IMG_SIZE, 3))
#model2 = applications.xception.Xception(weights = "imagenet", include_top=False, input_shape = (IMG_SIZE, IMG_SIZE, 3))
#model2 = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (IMG_SIZE, IMG_SIZE, 3))

for layer in model.layers[:5]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(14, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

history = model_final.fit(X_train, Y_train, epochs=20,batch_size=32, validation_data=(X_test, Y_test), verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

import pickle
def create_test_data(test_dir):
    if os.path.exists('testX.pickle'):
        with open('testX.pickle', 'rb') as handle:
            XTest = pickle.load(handle)
        return XTest
    XTest = defaultdict()
    for image in os.listdir(test_dir):
        img = cv2.imread(test_dir+"/"+image)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        XTest[image] = np.asarray(img)
    with open('testX.pickle', 'wb') as handle:
        pickle.dump(XTest, handle)
    return XTest
XTest = create_test_data("test_final")

#model_final.save(path+'/model_transfer3.h5')
# model = load_model(path+"/model_transfer3.h5")
model = model_final
img = []
label = []
for x in XTest:
    x_t = XTest[x] 
    x_t = np.expand_dims(x_t, axis=0)
    y = model.predict(x_t/255.)
    img.append(x)
    label.append(np.argmax(y[0])+1)
dic = {'id':img,'label':label}
df = pd.DataFrame(dic)
df.to_csv('submit_trans5.csv',index=False)

