import numpy as np
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
src=np.load('preprocessed data/src_train.npy')
dst=np.load('preprocessed data/dst_train.npy')
attacktype_label=np.load('preprocessed data/attacktype_label_train.npy')
#attackt_label=np.load('preprocessed data/attack_lbael.npy')
print(src.shape)
print(dst.shape)
print(attacktype_label.shape)
x_train=[]
for i in range(0,len(src)):
    temp=[]
    temp.append(src[i])
    temp.append(dst[i])
    x_train.append(temp)
x_train=np.array(x_train)
batch_size = 25
nb_classes = 38
def modelDense(x_train):
    model=Sequential()
    model.add(Dense(10,activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(14,activation='relu'))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model=modelDense(x_train)
model.fit(x_train,attacktype_label, batch_size=batch_size, epochs=10)