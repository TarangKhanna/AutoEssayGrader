from sklearn.externals import joblib
import pandas as pd 
# comment below for testing 
#"""
# from essaygrader.trainGrader import NumWordsTransformer
# from essaygrader.trainGrader import NumStopWordsTransformer
# from essaygrader.trainGrader import NumIncorrectSpellingTransformer
# from essaygrader.trainGrader import NumCharTransformer
# from essaygrader.trainGrader import NumPunctuationTransformer
# from essaygrader.trainGrader import NumIncorrectGrammarTransformer
# from essaygrader.trainGrader import NumIncorrectGrammarTransformer
## uncomment below for testing """

"""from trainGrader import NumWordsTransformer
from trainGrader import NumStopWordsTransformer
from trainGrader import NumIncorrectSpellingTransformer
from trainGrader import NumCharTransformer
from trainGrader import NumPunctuationTransformer
from trainGrader import NumIncorrectGrammarTransformer
from trainGrader import NumIncorrectGrammarTransformer"""

class predictGrades:
    def __init__(self):
        self.df = None
        pass

    # input = essay in text form
    # returns percentage of grade
    def predict(self, essay):
        clf = joblib.load('gradingModel.pkl')
        df = pd.DataFrame()
        df['essay'] = [essay]
        # print (df['essay'])
        confidence = (clf.predict_proba(df['essay'])[0][ord(clf.predict(df['essay'])[0]) - ord('A')])
        # print (confidence)
        return (clf.predict(df['essay'])[0], confidence)

if __name__ == "__main__":
    pg = predictGrades()
    print (pg.predict("THis be failin. sjjbfjbefbebkfleblkrjfbjkf..efef.ef´rƒe®ƒerffr/=."))