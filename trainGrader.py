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
    self.df = None
    pass

  def readData(self):
      file_name_training = 'original_training_data.xlsx'
      xl = pd.ExcelFile(file_name_training)
      print xl.sheet_names
      self.df = xl.parse("training_set")
      # print self.df.head()
      # we expect 1785 rows of training data, but found 1783
      # self.cleanData()
      return self.df.loc[self.df['essay_set'] == 1]
      
  # def cleanData(self):
  #   self.df.dropna()
  #   self.df[self.df['domain1_score'].apply(lambda x: str(x).isdigit())]
    
if __name__ == "__main__":
  predict = predictGrades()
  data = predict.readData()
  
  data.dropna()
  data[data['domain1_score'].apply(lambda x: str(x).isdigit())]
  data['domain1_score'] = data['domain1_score'].astype(int)

  # use essay set 1 for now, has 2-12 for grade range, convert this to 0 to 100%?
  essay_set = data['essay_set']
  print essay_set
  # X = essay data    
  # use essay_set to understand the context of the essay
  # deal with Anonymization in essay 
  essay = data['essay']
  print essay
  # Y = domain1_score, since all essays havbe this and it considers rater1 and rater2's score
  # need to normalize / clean this
  grade = data['domain1_score']
  print grade

  # trying out svm to get the accuracy
  clf = svm.SVC()
  clf.fit(essay, grade)

  clf.predict(essay[28])
  
# Datas columns descriptions:

# essay_id: A unique identifier for each individual student essay
# essay_set: 1-8, an id for each set of essays
# essay: The ascii text of a student's response
# rater1_domain1: Rater 1's domain 1 score; all essays have this
# rater2_domain1: Rater 2's domain 1 score; all essays have this
# rater3_domain1: Rater 3's domain 1 score; only some essays in set 8 have this.
# domain1_score: Resolved score between the raters; all essays have this
# rater1_domain2: Rater 1's domain 2 score; only essays in set 2 have this
# rater2_domain2: Rater 2's domain 2 score; only essays in set 2 have this
# domain2_score: Resolved score between the raters; only essays in set 2 have this
# rater1_trait1 score - rater3_trait6 score: trait scores for sets 7-8
