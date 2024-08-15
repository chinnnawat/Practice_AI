# import numpy as np
# from numpy.linalg import inv
# import pandas as pd
# import matplotlib.pyplot as plt

# # *********************************************************** Create Functions ***********************************************************

# # create Xb
# def MR_create_Xb(X):
#     N = X.shape[0]
#     ones = np.ones([N, 1])
#     Xb = np.hstack([ones, X])
#     return Xb

# # create W
# def MR_find_W(X, Y):
#     Xb = MR_create_Xb(X)
#     front = inv(np.dot(Xb.T, Xb))
#     back = np.dot(Xb.T, Y)
#     W = np.dot(front, back)
#     return W

# # create Predict
# def MR_predict(X, W):
#     Xb = MR_create_Xb(X)
#     Yhat = np.dot(Xb, W)
#     return Yhat

# # SSE (Sum of Squared Errors)
# def find_SSE(Y, Yhat):
#     SSE = np.sum((Y - Yhat)**2)
#     return SSE

# # MSE (Mean Squared Error)
# def find_MSE(Y, Yhat):
#     N = Y.shape[0]
#     SSE = ((Y - Yhat)**2).sum()
#     MSE = SSE / N
#     return MSE

# # MAE (Mean Absolute Error)
# def find_MAE(Y, Yhat):
#     N = Y.shape[0]
#     MAE = (np.abs(Y - Yhat)).sum() / N
#     return MAE

# # MAPE (Mean Absolute Percentage Error)
# def find_MAPE(Y, Yhat):
#     N = Y.shape[0]
#     MAPE = np.abs((Y - Yhat) / Y).sum()*100 / N
#     return MAPE

# def find_error(Y, Yhat, TypeOfError):
#     if TypeOfError == 'SSE':
#         error = find_SSE(Y, Yhat)
#     elif TypeOfError == 'MSE':
#         error = find_MSE(Y, Yhat)
#     elif TypeOfError == 'MAE':
#         error = find_MAE(Y, Yhat)
#     elif TypeOfError == 'MAPE':
#         error = find_MAPE(Y, Yhat)
#     return error


# # *********************************************************** read DATA ***********************************************************
# Data = pd.read_excel('data/BodyFatPercent.xlsx', usecols='A:O', skiprows=range(1,6))

# # change data to matrix
# DataMatrix = Data.values

# # *********************************************************** split DATA ***********************************************************

# # split matrix to Y, X
# D = DataMatrix.shape[1]-1

# Y = DataMatrix[:, D:]
# X = DataMatrix[:, :D]

# # Y train, X train 80% of data
# N = Y.shape[0]
# start = 0
# end = int(0.8*N)
# Y_train = Y[start:end, :]
# X_train = X[start:end, :]

# # Y test, X test 20% of data
# Y_test = Y[end:, :]
# X_test = X[end:, :]

# # *********************************************************** Train Model ***********************************************************
# W = MR_find_W(X_train, Y_train)

# # evaluation for training set
# Yhat_train = MR_predict(X_train, W)

# # prediction for test set
# Yhat_test = MR_predict(X_test, W)

# # *********************************************************** Test Model ***********************************************************
# # Check error
# # prediction for test set
# Yhat_test = MR_predict(X_test, W)
# error_test = find_error(Y_test, Yhat_test, 'MAPE')
# print(error_test)

# # *********************************************************** Result ***********************************************************
# # Real Train และ Predicted Train
# fig1 = plt.figure(figsize=(9, 4.5))
# plt.plot(Y_train, label='Real Train')
# plt.plot(Yhat_train, label='Predicted Train')
# plt.legend()
# # save figure
# plt.savefig('output1.png')

# # Scatter plot Y_train และ Yhat_train
# fig2 = plt.figure(figsize=(9, 4.5))
# plt.scatter(Yhat_train, Y_train)
# plt.xlabel('Predicted Train')
# plt.ylabel('Real Train')
# # save figure
# plt.savefig('output2.png')
