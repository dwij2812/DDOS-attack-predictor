import pandas
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
df= pandas.read_csv("../Data/corrected", header=None, names = col_names)

num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
features = df[num_features].astype(float)
att=list(df['label'])
label=list(set(att))

num=[]
num1=[]
for i in range(0,len(label)):
    for j in att:
        if(label[i]==j):
            num.append(i)

attacktype_label=np_utils.to_categorical(num,38)
for i in att:
    if(i=='normal.'):
        num1.append(1)
    else:
        num1.append(0)
attack_label=np_utils.to_categorical(num1,2)
src=np.array(df['src_bytes'])
dst=np.array(df['dst_bytes'])
src_train, src_test, dst_train, dst_test,attack_label_train,attack_label_test,attacktype_label_train,attacktype_label_test = train_test_split(src,dst,attack_label ,attacktype_label,test_size=0.33, random_state=42)
np.save('../preprocessed data/attack_label_train.npy',attack_label_train)
np.save('../preprocessed data/attacktype_label_train.npy',attacktype_label_train)
np.save('../preprocessed data/src_train.npy',src_train)
np.save('../preprocessed data/dst_train.npy',dst_train)
np.save('../preprocessed data/attack_label_test.npy',attack_label_test)
np.save('../preprocessed data/attacktype_label_test.npy',attacktype_label_test)
np.save('../preprocessed data/src_test.npy',src_test)
np.save('../preprocessed data/dst_test.npy',dst_test)