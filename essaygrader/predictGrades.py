from sklearn.externals import joblib
from essaygrader.trainGrader import NumWordsTransformer
from essaygrader.trainGrader import NumStopWordsTransformer
from essaygrader.trainGrader import NumIncorrectSpellingTransformer
from essaygrader.trainGrader import NumCharTransformer
from essaygrader.trainGrader import NumPunctuationTransformer
from essaygrader.trainGrader import NumIncorrectGrammarTransformer
from essaygrader.trainGrader import NumIncorrectGrammarTransformer
import pandas as pd 

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
        print (df['essay'])
        return clf.predict(df['essay'])[0]

if __name__ == "__main__":
    pg = predictGrades()
    print (pg.predict("random test"))