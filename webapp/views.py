from django.shortcuts import render
from django.http import HttpResponse
from .request_handler import render_to_populated_response, get_param_with_default
from .upload_essay_form import UploadEssayForm

from essaygrader.predictGrades import predictGrades
#from sklearn.base import TransformerMixin
from essaygrader.trainGrader import NumWordsTransformer
from django.template import Context, loader

def login(request):
    template = loader.get_template("./login.html")
    return HttpResponse(template.render())

def view_past_essays(request):

        return HttpResponse('Past essay page') #TODO

def contact_us(request):

        return HttpResponse('Contact Us Page') #TODO

def logout(request):

        return HttpResponse('Log out page') #TODO

def handle_uploaded_essay(essay_file):

    print("in handle uploaded essay")
    essay = ""
    for chunk in essay_file.chunks():
        essay += chunk.decode("utf-8")
    try:
        predictor = predictGrades()
        essay_grade = predictor.predict(essay)    
        print("Essay Grade: " + essay_grade)
    except:
        essay_grade = "Couldn't be determined.."

    return essay_grade

def submit_essay(request):

    print("In submit essay")
    if request.method == 'POST':
        form = UploadEssayForm(request.POST,request.FILES)
        if form.is_valid():
            essay_grade = handle_uploaded_essay(request.FILES['file'])
            return render_to_populated_response('index.html',\
            {'title':"Auto Essay Grader",\
            'user_name': "cs407",\
            'form': form,\
            'essay_grade':essay_grade},request)
        else:
            #This will return the same form but with errors displayed to the user
            return render_to_populated_response('index.html',\
            {'title':"Auto Essay Grader",\
            'user_name': "cs407",\
            'form': form},request)

    #If its not a POST request then mimic the upload essay page
    return upload_essay(request)

#Required keys: 'title' and 'user_name' for template.html that is used by all pages after log in/sign up
def upload_essay(request):

    form = UploadEssayForm()
    return render_to_populated_response('index.html',\
        {'title':"Auto Essay Grader",\
        'user_name': "cs407",\
        'form': form},request)

"""# custom scikit learn transformer to incorporate number of words
# of the essay into the features
class NumWordsTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        lengths = pd.DataFrame(X)
        lengths.sort_index(inplace=True)
        # convert words to lower?
        l = lengths['essay'].str.split(" ").str.len()
        print (l)
        return pd.DataFrame(l)
    def fit(self, X, y=None, **fit_params):
        return self
"""
