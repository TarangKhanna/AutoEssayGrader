from django.shortcuts import render
from django.http import HttpResponse
from .request_handler import render_to_populated_response, get_param_with_default
from .upload_essay_form import UploadEssayForm
from .login_form import UploadLoginForm

from essaygrader.predictGrades import predictGrades
from essaygrader.trainGrader import NumWordsTransformer
from django.template import Context, loader
from nltk.corpus import stopwords
import traceback
from nltk.tokenize import word_tokenize
from enchant.checker import SpellChecker


# from firebase import firebase
# firebase = firebase.FirebaseApplication('https://your_storage.firebaseio.com', None)
# result = firebase.get('/users', None)
# print result
# {'1': 'John Doe', '2': 'Jane Doe'}


def login(request):
    form = UploadLoginForm()
    return render_to_populated_response('login.html',\
        {'title':"Auto Essay Grader"},request)

def view_past_essays(request):

        return HttpResponse('Past essay page') #TODO

def contact_us(request):

        return HttpResponse('Contact Us Page') #TODO

def logout(request):
        # return render_to_populated_response('login.html',\
        #     {'title':"Auto Essay Grader"},request)
        return HttpResponse('Log out page') #TODO

def handle_uploaded_essay(essay_file):

    essay = ""
    essay_grade = ""
    word_count = 0
    spelling_error_count = 0
    stop_word_count = 0
    spelling_errors = []
    for chunk in essay_file.chunks():
        essay += chunk.decode("utf-8")
    try:
        predictor = predictGrades()
        essay_grade = predictor.predict(essay)
        word_count = get_word_count(essay)
        stop_word_count = get_stop_word_count(essay)
        spelling_error_count, spelling_errors = get_spelling_error_count(essay)

        print("Essay Grade: " + str(essay_grade) + " %")
    except Exception as e:
        print(e)
        traceback.print_exc()
        essay_grade = "Couldn't be determined.."

    return essay_grade,essay,word_count,stop_word_count,spelling_error_count, spelling_errors

def get_word_count(essay):
    return len(essay.split(" "))

def get_stop_word_count(essay):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(essay.lower())
    stop_word_count = len([w for w in word_tokens if w in stop_words])
    print("stop words: " + str(stop_words))
    return stop_word_count

def get_spelling_error_count(essay):
    spelling_error_count = 0
    spelling_errors = []
    chkr = SpellChecker("en_US")
    chkr.set_text(essay)
    for err in chkr:
        spelling_error_count += 1
        spelling_errors.append(err.word)

    return spelling_error_count, spelling_errors

def submit_essay(request):

    if request.method == 'POST':
        form = UploadEssayForm(request.POST,request.FILES)
        if form.is_valid():
            essay_grade,essay,word_count,stop_word_count,spelling_error_count, spelling_errors = handle_uploaded_essay(request.FILES['file'])
            return render_to_populated_response('upload_essay.html',\
            {'title':"Auto Essay Grader",\
            'user_name': "cs407",\
            'form': form,\
            'essay_grade':essay_grade,\
            'essay':essay,\
            'word_count':word_count,\
            'spelling_error_count':spelling_error_count,\
            'stop_word_count':stop_word_count,\
            'spelling_errors':spelling_errors},request)
        else:
            #This will return the same form but with errors displayed to the user
            return render_to_populated_response('upload_essay.html',\
            {'title':"Auto Essay Grader",\
            'user_name': "cs407",\
            'form': form},request)

    #If its not a POST request then mimic the upload essay page
    return upload_essay(request)

#Required keys: 'title' and 'user_name' for template.html that is used by all pages after log in/sign up
def upload_essay(request):

    form = UploadEssayForm()
    return render_to_populated_response('upload_essay.html',\
        {'title':"Auto Essay Grader",\
        'user_name': "cs407",\
        'form': form},request)
