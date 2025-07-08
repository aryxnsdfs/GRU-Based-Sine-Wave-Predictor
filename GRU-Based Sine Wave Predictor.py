import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,Dense
import matplotlib.pyplot as plt

x=np.arange(0,100,0.1)
data=np.sin(x)

def data_set(data,size):
    x,y=[],[]
    for i in range(len(data)-size):
        x.append(data[i:i+size])
        y.append(data[i+size])
    return np.array(x),np.array(y)

size=50
x,y=data_set(data,size)
x=x.reshape((x.shape[0],x.shape[1],1))

split=int(len(x)*0.8)
x_train,x_test=x[:split],x[split:]
y_train,y_test=y[:split],y[split:]

model=Sequential([
    GRU(64,input_shape=(size,1)),
    Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.fit(x_train,y_train,epochs=20,validation_split=0.2)

pred=model.predict(x_test)

plt.plot(y_test,label='true')
plt.plot(pred,label='predicted')
plt.legend()
plt.show()