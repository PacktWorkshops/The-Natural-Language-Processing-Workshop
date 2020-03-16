# coding: utf-8


from sklearn.metrics import mean_squared_error
from math import sqrt
y_actual = [0,1,2,1,0]
y_predicted = [0.03,1.2,1.6,.9,0.1]
rms = sqrt(mean_squared_error(y_actual, y_predicted))
print('Root Mean Squared Error (RMSE) is:', rms)
assert round(rms,1) == 0.2



from sklearn.metrics import mean_absolute_error
y_actual = [0,1,2,1,0]
y_predicted = [0.03,1.2,1.6,.9,0.1]
mape = mean_absolute_error(y_actual, y_predicted) * 100
print('Mean Absolute Percentage Error (MAPE) is:', round(mape,2), '%')
assert round(mape, 1) == 16.6
