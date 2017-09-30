from django.shortcuts import render
from django.http import HttpResponse
from .request_handler import render_to_populated_response, get_param_with_default
from essaygrader.trainGrader import predictGrades

def login(request):

	return HttpResponse('Log in page') #TODO

def view_past_essays(request):

		return HttpResponse('Past essay page') #TODO

def contact_us(request):

		return HttpResponse('Contact Us Page') #TODO

def logout(request):

		return HttpResponse('Log out page') #TODO

def upload_essay(request):

	#Required keys: 'title' and 'user_name' for template.html that is used by all pages after log in/sign up
    return render_to_populated_response('index.html',\
    {'title':"Auto Essay Grader",\
    'user_name': "cs407" },request)
