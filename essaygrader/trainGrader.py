from __future__ import division # preventing division issue in 2.7
import pandas as pd 
import math
import numpy as np
from textblob import TextBlob
from sklearn import svm
from sklearn.externals import joblib
from nltk.tokenize import RegexpTokenizer
from sklearn.base import TransformerMixin
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
import pylab
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
import enchant

# custom scikit learn transformer to incorporate number of words
# of the essay into the features
class NumWordsTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        l = lengths['essay'].str.split(" ").str.len()
        # print (l)
        return pd.DataFrame(l)

    def fit(self, X, y=None, **fit_params):
        return self

class NumStopWordsTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        lengths['stopWords'] = lengths.apply(self.stopWordHelper, axis=1)
        return pd.DataFrame(lengths['stopWords'])

    def fit(self, X, y=None, **fit_params):
        return self
    
    def stopWordHelper(self, row):
      count = 0
      s = set(stopwords.words('english'))
      for word in row['essay'].split(" "):
        if word in s:
          count += 1
      return count
    
class NumIncorrectSpellingTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        lengths['IncorrectSpelling'] = lengths.apply(self.spellCheckHelper, axis=1)
        return pd.DataFrame(lengths['IncorrectSpelling'])

    def fit(self, X, y=None, **fit_params):
        return self

    def spellCheckHelper(self, row):
        count = 0
        tokenizer = RegexpTokenizer(r'\w+')
        enchantDictionary = enchant.Dict("en_US")
        # use this tokenizer since it eliminates punctuation
        for word in tokenizer.tokenize(row['essay']):
          if not enchantDictionary.check(word):
            count += 1
        return count

class trainModel:
  def __init__(self):
    self.df = None
    pass

  def readData(self):
      file_name_training = 'original_training_data.xlsx'
      xl = pd.ExcelFile(file_name_training, options={'encoding':'utf-8'})
      # print(xl.sheet_names)
      self.df = xl.parse("training_set")
      # we expect 1785 rows of training data, but found 1783
      # self.cleanData()
      self.df.loc[self.df['essay_set'] == 1, 'domain1_score'] *= 100/12
      self.df.loc[self.df['essay_set'] == 3, 'domain1_score'] *= 100/3
      return self.df.loc[(self.df['essay_set'] == 1) | (self.df['essay_set'] == 3)]
      
  # def cleanData(self):
  #   self.df.dropna()
  #   self.df[self.df['domain1_score'].apply(lambda x: str(x).isdigit())]

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

if __name__ == "__main__":
    train = trainModel()
    data = train.readData()

    data.dropna()
    # data[data['domain1_score'].apply(lambda x: str(x).isdigit())]
    # data['domain1_score'] = data['domain1_score'].astype(int)

    # use essay set 1 for now, has 2-12 for grade range, convert this to 0 to 100%?
    essay_set = data['essay_set']
    # print(essay_set)
    # X = essay data    
    # use essay_set to understand the context of the essay
    # deal with Anonymization in essay 
    essay = data['essay']

    # data['text_length'] =  data['essay'].str.len()
    # print(data['text_length'])

    # data['word_length'] =  data['essay'].str.len()
    # print(data['word_length'])

    # print(essay)
    # Y = domain1_score, since all essays havbe this and it considers rater1 and rater2's score
    # need to normalize / clean this, scale min 2 , max 12 to min 0 max 100
    grade = data['domain1_score'].astype(int)
    print(grade)

    # text to vector 
    # essay = vectorizer.fit_transform(essay)

    # trying out svm to get the accuracy
    # convert to regression problem
    clf = svm.SVC()

    # X = essay,  data['text_length']

    # X_train, X_test, y_train, y_test = train_test_split(essay,grade,test_size=0.6)

    # switch to word2vec
    # add feature union to support multiple features
    # pipe_clf = Pipeline([('vect', CountVectorizer()), ('svm', MultinomialNB())])
    # pipe_clf = Pipeline([('vectorizer', CountVectorizer()), ('svm', clf)])

    pipe_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
        ('counts', CountVectorizer())
        ])),
        ('word_count', NumWordsTransformer()),
        ('num_stop_words', NumStopWordsTransformer())
    #   ('num_incorrect_spellings', NumIncorrectSpellingTransformer())
        ])),
        ('classifier', clf)
    ])

    # uncomment to train and save model 
    # pipe_clf.fit(X_train,y_train)
    # accuracy = pipe_clf.score(X_test,y_test)
    # print(accuracy)
    # joblib.dump(pipe_clf, 'gradingModel.pkl')

    

    # to find best hyper parameters--testing phase
    title = "Learning Curves (SVC, default)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(pipe_clf, title, essay, grade, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

    # print ('Prediction:')
    # print (pipe_clf.predict(essay[1000:1020]))

# Data columns descriptions:

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


# features to add:
# Number of exclamation marks
# Number of question marks
# Number of “difficult” words (vocab)
# Number of spelling mistakes (spelling).
# Number of stopwords