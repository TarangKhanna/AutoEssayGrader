from sklearn.externals import joblib
from essaygrader.trainGrader import NumWordsTransformer
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
        return round(((float) (clf.predict(df['essay'])[0])/12.0 * 100.0), 2)

if __name__ == "__main__":
    pg = predictGrades()
    print (pg.predict("random test"))