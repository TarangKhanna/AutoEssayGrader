from __future__ import division 
import pandas as pd 
import re, math
from collections import Counter
import string
import numpy as np
from sklearn.metrics import classification_report
# for SVM with rbf
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
import difflib
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
import language_check

# todo:
# 1) Convert to regression and train on more data, normaliza 0 to 1 and use kappa
# 2) Use A/B/C/D and keep doing classification-- done, 73% accuracy with KNN
# 3) essay & prompt cosine similarity by filtering by id, 
# or just add query as one column to the data

# custom scikit learn transformer to incorporate number of words
# of the essay into the features
class NumWordsTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        l = lengths['essay'].str.split(" ").str.len()
        return pd.DataFrame(l)

    def fit(self, X, y=None, **fit_params):
        return self

class NumCharTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        l = lengths['essay'].str.len()
        return pd.DataFrame(l)

    def fit(self, X, y=None, **fit_params):
        return self
    
class NumCharTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        l = lengths['essay'].str.len()
        return pd.DataFrame(l)

    def fit(self, X, y=None, **fit_params):
        return self

class NumPunctuationTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        lengths['punctuation'] = lengths.apply(self.punctuationHelper, axis=1)
        return pd.DataFrame(lengths['punctuation'])

    def fit(self, X, y=None, **fit_params):
        return self
    
    def punctuationHelper(self, row):
      count = 0
      s = set(string.punctuation)
      for word in row['essay'].split(" "):
        if word in s:
          count += 1
      return count

class NumIncorrectGrammarTransformer(TransformerMixin):
    tool = language_check.LanguageTool('en-US')
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        lengths['grammar'] = lengths.apply(self.grammarHelper, axis=1)
        return pd.DataFrame(lengths['grammar'])

    def fit(self, X, y=None, **fit_params):
        return self
    
    def grammarHelper(self, row):
      count = 0
      for sentence in row['essay'].split("."):
          matches = self.tool.check(sentence)
          count += len(matches)
      return count

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

# class PromptSimilarityTransformer(TransformerMixin):
#     def transform(self, X, **transform_params):
#         df = pd.DataFrame(X)
#         prompt = df['prompt']
#         essay = df['essay']
#         # print (l)
#         return pd.DataFrame(l)

#     def fit(self, X, y=None, **fit_params):
#         return self

class trainModel:
    def __init__(self):
        self.df = None

    def get_cosine(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def text_to_vector(self, text):
        WORD = re.compile(r'\w+')
        words = WORD.findall(text)
        return Counter(words)

    def readData(self): 
        file_name_training = 'original_training_data.xlsx'
        xl = pd.ExcelFile(file_name_training, options={'encoding':'utf-8'})
        # print(xl.sheet_names)
        self.df = xl.parse("training_set")

        # remove @CAPS , etc from essay 
        
        # we expect 1785 rows of training data, but found 1783
        # get percentages
        self.df.loc[self.df['essay_set'] == 1, 'domain1_score'] *= 100/12
        self.df.loc[self.df['essay_set'] == 8, 'domain1_score'] *= 100/60
        self.df.loc[self.df['essay_set'] == 3, 'domain1_score'] *= 100/3
        self.df.loc[self.df['essay_set'] == 4, 'domain1_score'] *= 100/3
        # convert percentages to grade cutoffs 

        self.df.loc[(self.df['domain1_score'] >= 35), 'domain1_grade'] = 'E'
        self.df.loc[(self.df['domain1_score'] >= 45), 'domain1_grade'] = 'D'
        self.df.loc[(self.df['domain1_score'] >= 55), 'domain1_grade'] = 'C'
        self.df.loc[(self.df['domain1_score'] >= 70), 'domain1_grade'] = 'B'
        self.df.loc[self.df['domain1_score'] >= 85, 'domain1_grade'] = 'A'
        self.df.loc[self.df['domain1_score'] < 35, 'domain1_grade'] = 'F'
        # preprocess and save this to csv

        # set the prompt
        prompt1 = """More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in 
        technology believe that computers have a positive effect on people. They teach hand-eye 
        coordination, give people the ability to learn about faraway places and people, and even 
        allow people to talk online with other people. Others have different ideas. Some experts are 
        concerned that people are spending too much time on their computers and less time exercising, 
        enjoying nature, and interacting with family and friends. Write a letter to your local newspaper
        in which you state your opinion on the effects computers have on people. 
        Persuade the readers to agree with you."""

        # calculate and store similarity
        self.df = self.df.loc[(self.df['essay_set'] == 1)]
        self.df_mod = self.df[['essay_set', 'essay']]
        cos_sims = []
        for row in self.df_mod.iterrows():
            # print (row[1]['essay_set'])
            if row[1]['essay_set'] == 1:
                s1 = row[1]['essay']
                s2 = prompt1
                s1w = re.findall('\w+', s1.lower())
                s2w = re.findall('\w+', s2.lower())
                s1cnt = Counter(s1w)
                s2cnt = Counter(s2w)
                common = set(s1w).intersection(s2w) 
                
                # common_ratio = difflib.SequenceMatcher(None, s1w, s2w).ratio()
                # print '%.1f%% of words common.' % (100*common_ratio)
                # cosine = self.get_cosine(vector1, vector2)
                # cos_sims.append(100*common_ratio)
                cos_sims.append(len(common)/len(prompt1))
                
        self.df['cosine'] = cos_sims

        # vector1 = text_to_vector(essay)
        # vector2 = text_to_vector(prompt1)
        print (self.df)
        # cosine = get_cosine(vector1, vector2)

        # print ('Cosine:', cosine)

        # self.df.loc[self.df['essay_set'] == 1, 'promptSimilarity'] = prompt1
        # self.df.loc[self.df['essay_set'] == 3, 'prompt'] = prompt3
        # self.df.loc[self.df['essay_set'] == 4, 'prompt'] = prompt4
        
        # c  | (self.df['essay_set'] == 3) | (self.df['essay_set'] == 4)
        return self.df

#   def cleanData(self):
#     self.df.dropna()
#     self.df[self.df['domain1_score'].apply(lambda x: str(x).isdigit())]

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
    print (data[['essay_id', 'domain1_score', 'domain1_grade', ]])
    # histogram to check distribution of grades
    data['domain1_grade'].value_counts().plot(kind='bar')
    # plt.show()
    data.dropna()
    
    # data.dropna(subset='vdomain1_score')
    # data set 4 seems to have infinite so we need this
    data = data[np.isfinite(data['domain1_score'])]
    # data[data['domain1_score'].apply(lambda x: str(x).isdigit())]
    # data['domain1_score'] = data['domain1_score'].astype(int)

    # use essay set 1 for now, has 2-12 for grade range, convert this to 0 to 100%?
    essay_set = data['essay_set']
    # print(essay_set)
    # X = essay data    
    # use essay_set to understand the context of the essay
    # deal with Anonymization in essay 
    essay = data['essay']
    # essay = data[['essay', 'prompt']]
    # data['text_length'] =  data['essay'].str.len()
    # print(data['text_length'])

    # data['word_length'] =  data['essay'].str.len()
    # print(data['word_length'])

    # print(essay)
    # Y = domain1_score, since all essays havbe this and it considers rater1 and rater2's score
    # need to normalize / clean this, scale min 2 , max 12 to min 0 max 100
    # grade = data['domain1_score'].astype(int)
    grade = data['domain1_grade']
    # print(grade)

    # text to vector 
    # essay = vectorizer.fit_transform(essay)

    # trying out svm to get the accuracy
    # convert to regression problem
    # clf = svm.SVR(kernel='poly', C=1e3, degree=2)
    # classification, with labels = 'A, B, C, D, E, F'
    # 67.8% accuracy with these parameters
    clf = svm.SVC(C=1, cache_size=500, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovo', gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=False,
    tol=0.001, verbose=False)

    # 71% accuracy with these parameters 
    # clf = KNeighborsClassifier(n_neighbors=100)

    # X = essay,  data['text_length']

    # scaler = StandardScaler()
    # X = scaler.fit_transform(essay)
    # X_2d = scaler.fit_transform(X)

    # switch to two inputs = (essay, prompt)
    X_train, X_test, y_train, y_test = train_test_split(essay,grade,test_size=0.5)

    # switch to word2vec
    # add feature union to support multiple features
    # pipe_clf = Pipeline([('vect', CountVectorizer()), ('svm', MultinomialNB())])
    # pipe_clf = Pipeline([('vectorizer', CountVectorizer()), ('svm', clf)])

    pipe_clf = Pipeline([
    ('features', FeatureUnion([
        ('ngram_tf_idf', Pipeline([
         ('counts', CountVectorizer()),
         ('scaler', StandardScaler(with_mean=False))
        ])),
        ('word_count', NumWordsTransformer()),
        ('char_count', NumCharTransformer()),
        ('num_stop_words', NumStopWordsTransformer()),
        ('num_punctuations', NumPunctuationTransformer())
        # ('prompt_similarity', PromptSimilarityTransformer())
        # ('num_incorrect_spellings', NumIncorrectSpellingTransformer())
        # ('num_grammar', NumIncorrectGrammarTransformer())
        ])),
        ('classifier', clf)
    ])

    # uncomment to train and save model 
    pipe_clf.fit(X_train,y_train)
    accuracy = pipe_clf.score(X_test,y_test)
    print(accuracy)
    joblib.dump(pipe_clf, 'gradingModel.pkl')

    y_true, y_pred = y_test, pipe_clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    # f1 score:
    # y_pred = pipe_clf.predict(X_test)
    # print (f1_score(y_test, y_pred, average='weighted'))

    # grid search:
    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # grid = GridSearchCV(clf, param_grid=param_grid, cv=None)
    # #  It is usually a good idea to scale the data for SVM training.
    # # scaler = StandardScaler()
    # # X = scaler.fit_transform(essay)
    # # X_2d = scaler.fit_transform(X)
    # vectorizer = CountVectorizer()
    # essay = vectorizer.fit_transform(essay)
    
    # grid.fit(essay, grade)

    # print("The best parameters are %s with a score of %0.2f"
    #     % (grid.best_params_, grid.best_score_))

    # validation curve:
    # title = "Learning Curves (SVC, default)"
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # plot_learning_curve(pipe_clf, title, essay, grade, (0.7, 1.01), cv=cv, n_jobs=4)

    # plt.show()

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