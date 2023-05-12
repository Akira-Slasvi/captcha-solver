from PIL import Image
import numpy as np
import copy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random


# folder chứa ảnh
path = 'D:\\MicroAuto_4_8_2019\\MicroAuto\\bin\\x86\\Release\\captchas'


xdata = []

ydata = []

xtrain = []

ytrain = []

xtest = []

ytest = []

files = []
# r=root, d=directories, f = files
cnt = 0
# load folder chứa ảnh. chia ảnh thành 4 phần với các nhãn
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))
random.shuffle(files)
print(len(files))
for file in files:
    if '.jpg' in file:
        im = np.array(Image.open(file).resize((128,36)).convert('1').convert('RGB'))     
        im1 = np.zeros((36,32, 1), dtype="float16")
        im2 = np.zeros((36,32, 1), dtype="float16")
        im3 = np.zeros((36,32, 1), dtype="float16")
        im4 = np.zeros((36,32, 1), dtype="float16")     
        for i in range(32):
            for j in range(36):
                if im[j][i][0] > 0:
                    im[j][i][0] = 255
                im1[j][i] = im[j][i][0] / 255.0
        for i in range(32,64):
            for j in range(36):
                if im[j][i][0] > 0:
                    im[j][i][0] = 255
                im2[j][i - 32] =  im[j][i][0]  / 255.0
        for i in range(64,96):
            for j in range(36):
                if im[j][i][0] > 0:
                    im[j][i][0] = 255
                im3[j][i - 64] =  im[j][i][0]  / 255.0
        for i in range(96,128):
            for j in range(36):
                if im[j][i][0] > 0:
                    im[j][i][0] = 255
                im4[j][i - 96] =  im[j][i][0] / 255.0
        xdata.append(im1)
        ydata.append(int(os.path.basename(file)[0]))
        xdata.append(im2)
        ydata.append(int(os.path.basename(file)[1]))
        xdata.append(im3)
        ydata.append(int(os.path.basename(file)[2]))
        xdata.append(im4)
        ydata.append(int(os.path.basename(file)[3]))
        cnt = cnt + 1
        if cnt % 10 == 0:
            print(cnt)

for i in range(round(len(xdata) * 0.9)):
    xtrain.append(xdata[i])
    ytrain.append(tf.one_hot(ydata[i],10))
for i in range(round(len(xdata) * 0.9), len(xdata)):
    xtest.append(xdata[i])
    ytest.append(tf.one_hot(ydata[i],10))

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xtest = np.array(xtest)
ytest = np.array(ytest)


model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(36,32,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='sigmoid'))
earlystop_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model.summary())

# train data
model.fit(xtrain, ytrain, callbacks=[earlystop_callback], validation_data=(xtest, ytest), batch_size=40, epochs=500, verbose=1)
# serialize model to JSON
model_json = model.to_json()
with open("D:\\model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("D:\\model3.h5")
print("Saved model to disk")
