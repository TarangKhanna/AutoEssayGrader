# mean reversion for long term prediction
from __future__ import division # preventing division issue in 2.7
import pandas as pd 
import math
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import time
import os
import datetime

class predictGrades:
  def __init__(self):
    pass

  def readData(self):
      file_name_training = 'original_training_data.xlsx'
      xl = pd.ExcelFile(file_name_training)
      print xl.sheet_names
      df = xl.parse("training_set")
      print df.head()

if __name__ == "__main__":
  predict = predictGrades()
  predict.readData()
  
  