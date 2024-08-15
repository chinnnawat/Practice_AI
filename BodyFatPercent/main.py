import pandas as pd
import matplotlib.pyplot as plt
from model_utils import MR_find_W, MR_predict
from error_metrics import find_error

# *********************************************************** read DATA ***********************************************************
Data = pd.read_excel('data/BodyFatPercent.xlsx', usecols='A:O', skiprows=range(1, 6))

# change data to matrix
DataMatrix = Data.values

# *********************************************************** split DATA ***********************************************************

# split matrix to Y, X
D = DataMatrix.shape[1] - 1

Y = DataMatrix[:, D:]
X = DataMatrix[:, :D]

# Y train, X train 80% of data
N = Y.shape[0]
start = 0
end = int(0.8 * N)
Y_train = Y[start:end, :]
X_train = X[start:end, :]

# Y test, X test 20% of data
Y_test = Y[end:, :]
X_test = X[end:, :]

# *********************************************************** Train Model ***********************************************************
W = MR_find_W(X_train, Y_train)

# evaluation for training set
Yhat_train = MR_predict(X_train, W)

# prediction for test set
Yhat_test = MR_predict(X_test, W)

# *********************************************************** Test Model ***********************************************************
error_test = find_error(Y_test, Yhat_test, 'MAPE')
print(f'Test MAPE: {error_test}')

# *********************************************************** Result ***********************************************************
# Real Train และ Predicted Train
fig1 = plt.figure(figsize=(9, 4.5))
plt.plot(Y_train, label='Real Train')
plt.plot(Yhat_train, label='Predicted Train')
plt.legend()
# save figure
plt.savefig('output1.png')

# Scatter plot Y_train และ Yhat_train
fig2 = plt.figure(figsize=(9, 4.5))
plt.scatter(Yhat_train, Y_train)
plt.xlabel('Predicted Train')
plt.ylabel('Real Train')
# save figure
plt.savefig('output2.png')