from django.shortcuts import render
from django.http import HttpResponse
from .request_handler import render_to_populated_response, get_param_with_default
from .upload_essay_form import UploadEssayForm
from .login_form import UploadLoginForm

from essaygrader.predictGrades import predictGrades
from essaygrader.trainGrader import NumWordsTransformer
from django.template import Context, loader
import traceback

def login(request):
    form = UploadLoginForm()
    return render_to_populated_response('login.html',\
        {'title':"Auto Essay Grader"},request)

def view_past_essays(request):

        return HttpResponse('Past essay page') #TODO

def contact_us(request):

        return HttpResponse('Contact Us Page') #TODO

def logout(request):

        return HttpResponse('Log out page') #TODO

def handle_uploaded_essay(essay_file):

    essay = ""
    for chunk in essay_file.chunks():
        essay += chunk.decode("utf-8")
    try:
        predictor = predictGrades()
        essay_grade = predictor.predict(essay)    
        print("Essay Grade: " + str(essay_grade) + " %")
    except Exception as e:
        print(e)
        traceback.print_exc()
        essay_grade = "Couldn't be determined.."

    return essay_grade,essay

def submit_essay(request):

    if request.method == 'POST':
        form = UploadEssayForm(request.POST,request.FILES)
        if form.is_valid():
            essay_grade,essay = handle_uploaded_essay(request.FILES['file'])
            return render_to_populated_response('upload_essay.html',\
            {'title':"Auto Essay Grader",\
            'user_name': "cs407",\
            'form': form,\
            'essay_grade':essay_grade,\
            'essay':essay},request)
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
