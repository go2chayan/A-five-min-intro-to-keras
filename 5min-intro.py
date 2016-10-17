# Authored by Md Iftekhar Tanveer (itanveer@cs.rochester.edu)
# May 22nd, 2016, 2:48 PM
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# # Generate data points
# ----------- Gaussian Data -----------
#class_1 = np.random.multivariate_normal(mean = (0.5,2),cov=[[3,1],[1,3]],size=500)
#class_2 = np.random.multivariate_normal(mean = (10,-8),cov=[[3,-1],[-1,3]],size=500)
# ------------- XOR Data --------------
# x,y = np.meshgrid(np.linspace(-1,1,40),np.linspace(-1,1,40))
# class_1 = np.hstack((x[x*y>0][None].T,y[x*y>0][None].T))
# class_2 = np.hstack((x[x*y<=0][None].T,y[x*y<=0][None].T))
# np.random.shuffle(class_1)
# np.random.shuffle(class_2)
# ----------- Disk-Sphare data ---------
x,y = np.meshgrid(np.linspace(-1,1,40),np.linspace(-1,1,40))
r = np.sqrt(x**2.+y**2.)
class_1 = np.hstack((x[r<2.5/3.][None].T,y[r<2.5/3.][None].T))
class_2 = np.hstack((x[r>2.5/3.][None].T,y[r>2.5/3.][None].T))
np.random.shuffle(class_1)
np.random.shuffle(class_2)


# Train-Test split
X_train = np.concatenate((class_1[:300,:],class_2[:300,:]),axis=0)
Y_train = np.concatenate((np.ones(300,dtype=int),np.zeros(300,dtype=int)),axis=0)
X_test = np.concatenate((class_1[300:400,:],class_2[300:400,:]),axis=0)
Y_test = np.concatenate((np.ones(100,dtype=int),np.zeros(100,dtype=int)),axis=0)
# Shuffle
i = np.random.permutation(len(X_train))
X_train = X_train[i,:]
Y_train = Y_train[i]
i = np.random.permutation(len(X_test))
X_test = X_test[i,:]
Y_test = Y_test[i]

# Display
plt.figure(figsize=(5,8))
plt.subplot(211)
plt.scatter(X_train[:,0],X_train[:,1],c=Y_train,cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.axis=('equal')
plt.subplot(212)
plt.scatter(X_test[:,0],X_test[:,1],c=Y_test,cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data')
plt.axis=('equal')
plt.subplots_adjust(top=0.96,hspace=0.26)
plt.show()

# =============== Keras =================
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils.np_utils import to_categorical

# Configure the NN
model = Sequential()
# Only the first layer needs input dimension
model.add(Dense(output_dim=3,input_dim=2))
model.add(Activation('relu'))
model.add(Dense(output_dim=3,activation='relu'))
model.add(Dense(output_dim=2,activation='relu'))

# Compile the model
model.compile(loss='mean_squared_error',\
     optimizer='SGD', metrics=['accuracy'])

# Iterate on training data
y_tr = to_categorical(Y_train)
y_tst = to_categorical(Y_test)
model.fit(X_train,y_tr,nb_epoch=100,batch_size=4)

# Evaluate the performance
eval_ = model.evaluate(X_test,y_tst,batch_size=4)
print eval_

# Print a model summary
model.summary()

# prediction on new data
classes = model.predict_classes(X_test,batch_size=4)
print classes

# Print model weights
print model.get_weights()

# ================ Display ================
plt.figure('test')
plt.scatter(X_test[:,0],X_test[:,1],c=classes,cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Classification Results')
plt.show()

# === Draw decision hyperplane in high-res ===
x,y = np.meshgrid(np.linspace(-1,1,1000),np.linspace(-1,1,1000))
data = np.hstack((x.reshape(1000000,1),y.reshape(1000000,1)))
print data.shape
classlabel = model.predict_classes(data).reshape(1000,1000).astype(np.float32)
plt.imshow(classlabel)
plt.show()




