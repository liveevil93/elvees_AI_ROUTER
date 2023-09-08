#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Dataset CSE-CIC-IDS2018 ###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, classification_report

df = pd.read_csv('./archive/02-21-2018.csv')
df


# In[ ]:


## Dropping correlation greater than 0.98 ##

y = df[['Label']]
X = df
X.drop('Label', axis=1, inplace=True)
X.drop('Timestamp', axis=1, inplace = True)

X.drop('Dst Port', axis=1, inplace = True)

dropping = ['ACK Flag Cnt', 'TotLen Fwd Pkts', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Std', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Fwd Header Len', 'Bwd Header Len', 'Pkt Len Min', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ECE Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Idle Std', 'Idle Max']

X.drop(dropping, axis=1, inplace = True)

X = X.dropna('columns')# drop columns with NaN
X = X[[col for col in X if X[col].nunique() > 1]]# keep columns where there are more than 1 unique values


# In[ ]:


## Mapping Normal as 0 and Attack as 1 ##

lmap = {'Benign' : 0, 'DDOS attack-LOIC-UDP' : 1, 'DDOS attack-HOIC' : 1}
y['Label'] = y['Label'].map(lmap)


# In[ ]:


## Splitting dataset to 67.5% Train, 22.5% Test, 10% Validation ##

sc = MinMaxScaler()
X = sc.fit_transform(X)

percen_90_X, X_val, percen_90_y, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)
print(percen_90_X.shape, X_val.shape)
print(percen_90_y.shape, y_val.shape)

percen_90_X = sc.fit_transform(percen_90_X)

X_train, X_test, y_train, y_test = train_test_split(percen_90_X, percen_90_y, test_size = 0.25, random_state = 42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


## Libs for model ##

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv2D, Conv1D, AveragePooling2D, Activation, Flatten, Dropout, Input, BatchNormalization, MaxPooling1D
from keras.layers import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)

import tensorflow as tf


# In[ ]:


## CNN model ##

inputs = Input(shape=(35,1), name='input')
x = Conv1D(32, 3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(10, activation='relu', name='dense_1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='relu', name='dense_2')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='relu', name='dense_3')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid', name='dense_4')(x)

cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
cnn_model.summary()


# In[ ]:


## Model training ##

cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=512)


# In[ ]:


## Final evaluation of the model ##

scores = cnn_model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


## Confusion matrix and metrix ##

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
#
# Get the predictions
#
y_pred = cnn_model.predict(X_test)
y_pred = y_pred.flatten()

y_pred = np.where(y_pred > 0.5, 1, 0)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


### Dataset NSL-KDD ###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

df = pd.read_csv('./NSL_KDD/KDDTrain+.csv', sep=',', names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficult'])
df


# In[ ]:


## Dropping correlation greater than 0.98 ##

df.drop('difficult', axis = 1, inplace = True)
df = df.dropna('columns')# drop columns with NaN
df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values

df.drop('dst_host_srv_serror_rate', axis = 1, inplace = True)
df.drop('srv_serror_rate', axis = 1, inplace = True)
df.drop('srv_rerror_rate', axis = 1, inplace = True)
df.drop('num_root', axis = 1, inplace = True)


# In[ ]:


## Mapping objects to int ##

pmap = { 'tcp' : 0, 'udp' : 1, 'icmp' : 2}
df['protocol_type'] = df['protocol_type'].map(pmap)

df['service'] = pd.Categorical(df['service'])
df['service'] = df['service'].cat.codes

df['flag'] = pd.Categorical(df['flag'])
df['flag'] = df['flag'].cat.codes


# In[ ]:


## Mapping Normal as 0, Attack as 1 ##

lmap = {'normal':0, 'neptune':1, 'warezclient':1, 'ipsweep':1, 'portsweep':1,
       'teardrop':1, 'nmap':1, 'satan':1, 'smurf':1, 'pod':1, 'back':1,
       'guess_passwd':1, 'ftp_write':1, 'multihop':1, 'rootkit':1,
       'buffer_overflow':1, 'imap':1, 'warezmaster':1, 'phf':1, 'land':1,
       'loadmodule':1, 'spy':1, 'perl':1}
df['class'] = df['class'].map(lmap)


# In[ ]:


## Splitting to X and y (labels of traffic type) ##

y = df[['class']]
X = df
X.drop('class', axis=1, inplace=True)


# In[ ]:


## Splitting dataset to 67.5% Train, 22.5% Test, 10% Validation ##

sc = MinMaxScaler()
X = sc.fit_transform(X)

percen_90_X, X_val, percen_90_y, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)
print(percen_90_X.shape, X_val.shape)
print(percen_90_y.shape, y_val.shape)

percen_90_X = sc.fit_transform(percen_90_X)

X_train, X_test, y_train, y_test = train_test_split(percen_90_X, percen_90_y, test_size = 0.25, random_state = 42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


## Libs for model ##

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv2D, Conv1D, AveragePooling2D, Activation, Flatten, Dropout, Input, BatchNormalization, MaxPooling1D
from keras.layers import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)


# In[ ]:


## CNN model ver. 1 ##

inputs = Input(shape=(36,1), name='input')
x = Conv1D(64, 3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(16, 3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(10, activation='relu', name='dense_1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='relu', name='dense_2')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='relu', name='dense_3')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid', name='dense_4')(x)

cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
cnn_model.summary()


# In[ ]:


## Model training ##

cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=512)


# In[ ]:


## Final evaluation of the model ##

scores = cnn_model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


## Confusion matrix and metrix ##

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
#
# Get the predictions
#
y_pred = cnn_model.predict(X_test)
y_pred = y_pred.flatten()

y_pred = np.where(y_pred > 0.5, 1, 0)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# In[ ]:





# In[ ]:


## CNN model ver. 2 ##

inputs = Input(shape=(36,1), name='input')
x = Conv1D(32, 3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(10, activation='relu', name='dense_1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='relu', name='dense_2')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='relu', name='dense_3')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid', name='dense_4')(x)

cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
cnn_model.summary()


# In[ ]:


## Model training ##

cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=512)


# In[ ]:


## Final evaluation of the model ##

scores = cnn_model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


## Confusion matrix and metrix ##

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
#
# Get the predictions
#
y_pred = cnn_model.predict(X_test)
y_pred = y_pred.flatten()

y_pred = np.where(y_pred > 0.5, 1, 0)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


# In[ ]:





# In[ ]:


### Multiclass NSL-KDD ###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

df = pd.read_csv('./NSL_KDD/KDDTrain+.csv', sep=',', names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'class', 'difficult'])
df


# In[ ]:


## Dropping correlation greater than 0.98 ##

df.drop('difficult', axis = 1, inplace = True)
df = df.dropna('columns')# drop columns with NaN
df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values

df.drop('dst_host_srv_serror_rate', axis = 1, inplace = True)
df.drop('srv_serror_rate', axis = 1, inplace = True)
df.drop('srv_rerror_rate', axis = 1, inplace = True)
df.drop('num_root', axis = 1, inplace = True)


# In[ ]:


## Mapping objects to int ##

pmap = { 'tcp' : 0, 'udp' : 1, 'icmp' : 2}
df['protocol_type'] = df['protocol_type'].map(pmap)

df['service'] = pd.Categorical(df['service'])
df['service'] = df['service'].cat.codes

df['flag'] = pd.Categorical(df['flag'])
df['flag'] = df['flag'].cat.codes


# In[ ]:


## Mapping Normal as 0, Attack as unique types of attack ##

lmap = {'normal':0, 'neptune':1, 'warezclient':2, 'ipsweep':3, 'portsweep':4,
       'teardrop':5, 'nmap':6, 'satan':7, 'smurf':8, 'pod':9, 'back':10,
       'guess_passwd':11, 'ftp_write':12, 'multihop':13, 'rootkit':14,
       'buffer_overflow':15, 'imap':16, 'warezmaster':17, 'phf':18, 'land':19,
       'loadmodule':20, 'spy':21, 'perl':22}
df['class'] = df['class'].map(lmap)


# In[ ]:


## Splitting to X and y (labels of traffic type) ##

y = df[['class']]
X = df
X.drop('class', axis=1, inplace=True)


# In[ ]:


## Splitting dataset to 67.5% Train, 22.5% Test, 10% Validation ##

sc = MinMaxScaler()
X = sc.fit_transform(X)

percen_90_X, X_val, percen_90_y, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)
print(percen_90_X.shape, X_val.shape)
print(percen_90_y.shape, y_val.shape)

percen_90_X = sc.fit_transform(percen_90_X)

X_train, X_test, y_train, y_test = train_test_split(percen_90_X, percen_90_y, test_size = 0.25, random_state = 42)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


## One-hot coding and reshaping y ##

y_train = tf.one_hot(y_train, depth = 23)
y_test = tf.one_hot(y_test, depth = 23)
y_val = tf.one_hot(y_val, depth = 23)

y_train = np.reshape(y_train, (-1,23))
y_test = np.reshape(y_test, (-1,23))
y_val = np.reshape(y_val, (-1,23))


# In[ ]:


## Libs for model ##

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv2D, Conv1D, AveragePooling2D, Activation, Flatten, Dropout, Input, BatchNormalization, MaxPooling1D
from keras.layers import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)


# In[ ]:


## CNN for multiclass ##

inputs = Input(shape=(36,1), name='input')
x = Conv1D(32, 3, activation='relu')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(10, activation='relu', name='dense_1')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='relu', name='dense_2')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='relu', name='dense_3')(x)
x = Dropout(0.5)(x)
outputs = Dense(23, activation='softmax', name='dense_4')(x)

cnn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
cnn_model.summary()


# In[ ]:


## Model training ##

cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=512)


# In[ ]:


## Final evaluation of the model ##

scores = cnn_model.evaluate(X_val, y_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


## Confusion matrix and metrix ##

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
#
# Get the predictions
#
y_pred = cnn_model.predict(X_test)
y_pred=np.argmax(y_pred, axis=1)
y_test_pred=np.argmax(y_test, axis=1)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test_pred, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(50, 50))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print('Accuracy: %.4f' % accuracy_score(y_test_pred, y_pred))
print('Precision: %.4f' % precision_score(y_test_pred, y_pred, average = 'macro'))
print('Recall: %.4f' % recall_score(y_test_pred, y_pred, average = 'macro'))
print('F1 Score: %.4f' % f1_score(y_test_pred, y_pred, average = 'macro'))


# In[ ]:




