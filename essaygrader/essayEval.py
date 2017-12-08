import pandas as pd 
import re, math
from collections import Counter

class essayEval:
    def __init__(self):
        file_name_training = 'original_training_data.xlsx'
        xl = pd.ExcelFile(file_name_training, options={'encoding':'utf-8'})
        self.df = xl.parse("training_set")
        self.df = self.df[['essay_id', 'essay_set', 'essay', 'domain1_score']]

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
    
    # return (essay, similarity) that it matches with else returns None
    def checkPlagiarism(self, essay_new):
        for row in self.df.iterrows():
            s1 = row[1]['essay']
            # print (s1)
            vector1 = self.text_to_vector(essay_new)
            vector2 = self.text_to_vector(s1)
            cosine = self.get_cosine(vector1, vector2)
            if cosine > 0.9:
                # print ('found similar')
                return (s1, cosine)
        return (None,None)
    
    # updates training data with new grade
    def updateTrainingDatabase(self, essay_new, grade_new):
        ind = len(self.df) 
        self.df.loc[ind] = [ind, 9, essay_new, grade_new]
        writer = pd.ExcelWriter('new_training_data.xlsx')
        self.df.to_excel(writer,'Sheet1')
        writer.save()

if __name__ == "__main__":
    ee = essayEval()
    # print (ee.checkPlagiarism('Dear learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - theres a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, its a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1 youll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening.'))
    ee.updateTrainingDatabase('Dear learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - theres a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, its a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1 youll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening.', 9)