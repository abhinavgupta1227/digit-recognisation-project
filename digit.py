from tensorflow.keras.datasets import mnist #built-in data set
from tensorflow.keras.models import Sequential #neural network layer by layer
from tensorflow.keras.layers import Dense #fully connected layer, flatten layer
from tensorflow.keras.utils import to_categorical #one-hot vector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

print("shape of training data: " ,x_train.shape)
print("shape of training lables: ", y_train.shape)

index=12
print("Lable: ", y_train[index])
plt.imshow(x_train[index], cmap='Greys')
plt.show()
# --- Step 3: Preprocess the data ---
# Flatten + normalize images
x_train = x_train.reshape(x_train.shape[0], 28*28) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28*28) / 255.0
print("After reshaping: ", x_train.shape, x_test.shape)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("After one-hot encoding: ", y_train[0])

#step 4: training and building model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)), #hidden layer
    Dense(10, activation='softmax') #output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
okok 