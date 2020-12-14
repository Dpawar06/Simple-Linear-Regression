ca# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:22:43 2020

@author: Deepak
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
from sklearn import linear_model

Del = pd.read_csv('H:/DATA SCIENCE/Modules/Module 6 Simpal linear regression/DataSets/delivery_time.csv')
Del
# 1st moment business Decision
Del.mean()
Del.mode()
Del.median()

# 2nd moment business Decision
np.std(Del['Weightgainedgrams'])
np.std(Del['CaloriesConsumed'])
np.var(Del['Weightgainedgrams'])
np.var(Del['CaloriesConsumed'])

range =  max(Del['Weightgainedgrams'])-min(Del['Weightgainedgrams'])
range
range =  max(Del['CaloriesConsumed'])-min(Del['CaloriesConsumed'])
range 
# 3rd moment business Decision Skewness
Del.skew()
Del["Weightgainedgrams"].skew() # Skew is +ve  mean > median  
Del['CaloriesConsumed'].skew()

# 4th moment business Decision Kurtosis
Del.kurt()
Del["Weightgainedgrams"].kurt() # 
Del['CaloriesConsumed'].kurt()

 # using sklearn library
# %matplotlib inline
# Visulize data
plt.xlabel("Delivery")
plt.ylabel("Sorting time")
plt.scatter(Del.SortingTime,  Del.DeliveryTime, color = 'red', marker = '+')
np.corrcoef(Del.SortingTime, Del.DeliveryTime)

new_Del = Del.drop('SortingTime', axis=1)
new_Del

from sklearn.linear_model import LinearRegression
# building a model
model = linear_model.LinearRegression()
model.fit(new_Del, Del.SortingTime)
model.predict([[4]]) # X
model.coef_ # m
model.intercept_  # b   m*x + b
# m*x + b  
# 1.64*4+6.582
new_Del

df = del.predict
pred = model.predict(new_Del)

new_Del[""] = pred
new_Del



# using statsmodel library
#model 1
import statsmodels.formula.api
model = smf.ols ('DeliveryTime ~ SortingTime',data= Del).fit()
model.summary()
model.coef_
model.intercept_

pred1 = model.predict(pd.DataFrame(Del['SortingTime'])) # predict the y
pred1

print (model.conf_int(0.05)) # 95% confidance interval
# intercept b = 2.97, slop m = 1.108, X = 10
1.108*+2.97
res = Del.DeliveryTime - pred1
sqres = res*res
mse = np.mean(sqres)
rmse = np.sqrt(mse) # RMSE value 


########## log Transformation ######
#### x = log(SortingTime)
plt.scatter(x = np.log(Del['SortingTime']), y= Del['DeliveryTime'], color = 'red')
np.corrcoef(np.log(Del.SortingTime),Del.DeliveryTime)

model2 = smf.ols('DeliveryTime ~ np.log(SortingTime)', data= Del).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(np.log(Del['SortingTime'])))
pred2

print(model2.conf_int(0.05)) # 95% confidence interval

res2 = Del.DeliveryTime - pred2
sqres2 = res*res
mse2 = np.mean(sqres2)
rmse2 = np.sqrt(mse2)

########## Exponential Transfomation #############
plt.scatter(x= (Del['SortingTime']), y = np.log(Del['DeliveryTime']))
np.corrcoef( Del.SortingTime, np.log(Del.DeliveryTime))

model3= smf.ols('np.log(DeliveryTime) ~ SortingTime' , data = Del).fit()
model3.summary()

pred_log = model3.predict(pd.DataFrame(Del['SortingTime']))
pred_log
pred3 = np.exp(pred_log)
pred3

print(model3.conf_int(0.05))

res3 = Del.DeliveryTime - pred_log
sqres3 = res*res
mse3 = np.mean(sqres3)
rmse3 = np.sqrt(mse3)

################ Polyinomial model  ###########

plt.scatter(x = np.log(Del['SortingTime']) , y = np.log(Del['DeliveryTime']))
np.corrcoef(np.log(Del.SortingTime), np.log(Del.DeliveryTime))
model4 = smf.ols('np.log(DeliveryTime)~ np.log(SortingTime)', data = Del).fit()
model4.summary()
pred_log = model4.predict(pd.DataFrame(np.log(Del['SortingTime'])))
pred_log
print(model4.conf_int(0.05))
res4 = Del.DeliveryTime -pred_log
sqres4 = res*res
mes4 = np.mean(sqres4)
rmse4 = np.sqrt(mes4)
























