import pandas as pd
import numpy as np
import math
from word2number import w2n
from sklearn import linear_model


dataFrames = pd.read_csv("/hiring.csv")
dataFrames.experience.fillna(0,inplace = True)#fill in place 0

#then fill in median
dataFrames["test_score(out of 10)"].fillna(dataFrames["test_score(out of 10)"].median(),inplace = True)
for i in dataFrames["experience"]:
  if(i == 0):
    continue
  num = w2n.word_to_num(i)
  dataFrames.replace(i,num,inplace = True)
#using a linear regression on model
linear = linear_model.LinearRegression()
linear.fit(dataFrames[["experience","test_score(out of 10)","interview_score(out of 10)"]],dataFrames["salary($)"])
#predict for 2years nine test score and 6 interview
linear.predict([[2,9,6]])
