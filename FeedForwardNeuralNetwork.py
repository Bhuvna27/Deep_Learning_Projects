import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

all_data = pd.read_csv("https://tinyurl.com/y2qmhfsr")

all_inputs = (all_data.iloc[:,0:3].values/255.0)
all_outputs = all_data.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(all_inputs,all_outputs,test_size = 1/3)
n = X_train.shape[0]


w_hidden = np.random.rand(3,3)
w_output = np.random.rand(1,3)

b_hidden = np.random.rand(3,1)
b_output = np.random.rand(1,1)

## Activation functions

relu = lambda x: np.maximum(x,0)
logistic = lambda x: 1/(1+np.exp(-x))


## Runs output through the neural network to get predicted outcomes
def forward_prop(X):
   
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1,A1, Z2, A2


## Calculate test accuracy 
test_predictions = forward_prop(X_test.transpose())[3] # grab only output layer, A2
test_comparisions = np.equal((test_predictions>= .5).flatten().astype(int),Y_test)

accuracy = sum(test_comparisions.astype(int)/X_test.shape[0])
print("Accuracy:",accuracy)



