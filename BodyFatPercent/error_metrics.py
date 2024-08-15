import numpy as np

# SSE (Sum of Squared Errors)
def find_SSE(Y, Yhat):
    SSE = np.sum((Y - Yhat)**2)
    return SSE

# MSE (Mean Squared Error)
def find_MSE(Y, Yhat):
    N = Y.shape[0]
    SSE = ((Y - Yhat)**2).sum()
    MSE = SSE / N
    return MSE

# MAE (Mean Absolute Error)
def find_MAE(Y, Yhat):
    N = Y.shape[0]
    MAE = (np.abs(Y - Yhat)).sum() / N
    return MAE

# MAPE (Mean Absolute Percentage Error)
def find_MAPE(Y, Yhat):
    N = Y.shape[0]
    MAPE = np.abs((Y - Yhat) / Y).sum()*100 / N
    return MAPE

def find_error(Y, Yhat, TypeOfError):
    if TypeOfError == 'SSE':
        error = find_SSE(Y, Yhat)
    elif TypeOfError == 'MSE':
        error = find_MSE(Y, Yhat)
    elif TypeOfError == 'MAE':
        error = find_MAE(Y, Yhat)
    elif TypeOfError == 'MAPE':
        error = find_MAPE(Y, Yhat)
    return error
