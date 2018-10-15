import numpy as np
from keras.models import model_from_json
from keras.models import load_model
src=np.load('../preprocessed data/src_test.npy')
dst=np.load('../preprocessed data/dst_test.npy')
attacktype_label=np.load('../preprocessed data/attack_label_test.npy')
x_test=[]
for i in range(0,len(src)):
    temp=[]
    temp.append(src[i])
    temp.append(dst[i])
    x_test.append(temp)
x_test=np.array(x_test)
model=load_model("../model/CNN.h5")
pred=model.evaluate(x=x_test,y=attacktype_label,verbose=1)
print("accuracy",pred[1])
print("loss",pred[0])