from sklearn.externals import joblib

class predictGrades:
    def __init__(self):
        self.df = None
        pass

    def predict(self, essay):
        clf = joblib.load('gradingModel.pkl')
        return clf.predict([essay])[0]

if __name__ == "__main__":
    pg = predictGrades()
    print (pg.predict("random test"))