from django.shortcuts import render
from django.http import HttpResponse
from .request_handler import render_to_populated_response, get_param_with_default
from .upload_essay_form import UploadEssayForm
from .login_form import UploadLoginForm
from .upload_essay_folder_form import UploadEssayFolderForm
from .GradedEssay import GradedEssay

from essaygrader.predictGrades import predictGrades
from essaygrader.trainGrader import NumWordsTransformer
from django.template import Context, loader
from nltk.corpus import stopwords
import traceback
from nltk.tokenize import word_tokenize
from enchant.checker import SpellChecker

from .firebase_util import auth, db, firebase

import json

import language_check

from PIL import Image

from nltk import tokenize
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from io import BytesIO

import base64
import os

def signup(request):

    if request.method == 'POST':
        form = UploadLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["username"]
            password = form.cleaned_data["password"]

            try:
                user = auth.create_user_with_email_and_password(email, password)
                request.session['user'] = user
                form = UploadEssayForm()
                return render_to_populated_response('upload_essay.html',\
                    {'title':"Auto Essay Grader",\
                    'user_name': email,\
                    'form': form,\
                    'success': "Sign Up Succesful"},request)

            except Exception as e:
                error_json = e.args[1]
                error = "Error: " + json.loads(error_json)['error']['message']
                form = UploadLoginForm()
                return render_to_populated_response('signup.html',\
                    {'title':"Auto Essay Grader",\
                    'form': form,\
                    'error':error},request)

    else:
        form = UploadLoginForm()
        return render_to_populated_response('signup.html',\
            {'title':"Auto Essay Grader",\
            'form': form},request)

def login(request):

    if request.method == 'POST':
        form = UploadLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["username"]
            password = form.cleaned_data["password"]

            try:
                user = auth.sign_in_with_email_and_password(email, password)
                request.session['user'] = user
                form = UploadEssayForm()
                return render_to_populated_response('upload_essay.html',\
                    {'title':"Auto Essay Grader",\
                    'user_name': email,\
                    'form': form,\
                    'success': "Login Succesful"},request)

            #Throws exception if sign in fails
            except Exception as e:
                error_json = e.args[1]
                error = "Error: " + json.loads(error_json)['error']['message']
                form = UploadLoginForm()
                return render_to_populated_response('login.html',\
                    {'title':"Auto Essay Grader",\
                    'form': form,\
                    'error':error},request)

            
        
    else:  
        form = UploadLoginForm()
        return render_to_populated_response('login.html',\
            {'title':"Auto Essay Grader",\
            'form': form},request)

def view_past_essays(request):

    if 'user' in request.session:

        essay_list = [] #List of dictionaries
        error = ""
        user_email = ""
        try:
            user_email = request.session['user']['email']
            user_id_token = request.session['user']['idToken']
            all_essays_by_email = db.child("user_essay").order_by_child("email").equal_to(user_email).get()
            #print("\n\nAll Essays By Email: ")
            for essay in all_essays_by_email.each():
                essay_list.append(essay.val())
        except Exception as e:
                error_json = e.args[1]
                error = "Error: " + json.loads(error_json)['error']['message']
                essay_list = [] 

        #print(str(essay_list))

        return render_to_populated_response('past_essays.html',\
        {'title':"Auto Essay Grader",\
        'error':error,\
        'essay_list':essay_list,\
        'essay_count':len(essay_list),\
        'user_name':user_email}, request)

    #Not Logged In
    else:
        return not_logged_in_redirect(request)


def not_logged_in_redirect(request):
    form = UploadLoginForm()
    return render_to_populated_response('login.html',\
        {'title':"Auto Essay Grader",\
        'form': form,\
        'error': "Please login to access the secured parts of this website!"},request)



def logout(request):

    if 'user' in request.session:
        del request.session['user']
        form = UploadLoginForm()
        return render_to_populated_response('login.html',\
            {'title':"Auto Essay Grader",\
            'form': form,\
            'success':"Logout Successful"},request)
    else:
        return not_logged_in_redirect(request)

def save_essay_to_db(essay_text, essay_grade, essay_title, request):

    if 'user' in request.session:
        user_name = request.session['user']['email']
        user_id_token = request.session['user']['idToken']

        try:
            # Get a reference to the database service
            db = firebase.database()
            data = {
                "email":user_name,\
                "title":essay_title,\
                "essay_text":essay_text,\
                "essay_grade":essay_grade
            }

            # Pass the user's idToken to the push method
            results = db.child("user_essay").push(data, user_id_token)
            return 1
        except Exception as e:
            print(e)
            traceback.print_exc()
            return 0

    return 0

def get_grammar_issues_html(grammar_issues_list):

    gil_html = ""
    for gi in grammar_issues_list:
        gil_html += "<pre>" + str(gi) + "</pre>"

    return gil_html

def get_spelling_error_html(spelling_errors):

    sel_html = ""

    for se in spelling_errors:
        sel_html += str(se) + "<br>"

    return sel_html

def handle_uploaded_folder(valid_files, essay_title, save_to_db, request):
    
    essays = []
    predictor = predictGrades()

    for essay_file in valid_files:
        essay = ""
        for chunk in essay_file.chunks():
            essay += chunk.decode("utf-8")

        try:
            essay_grade, confidence_score = predictor.predict(essay)
            confidence_score = int((round(confidence_score,2) * 100)) #Percentage
            word_count = get_word_count(essay)
            stop_word_count = get_stop_word_count(essay)
            spelling_error_count, spelling_errors = get_spelling_error_count(essay)
            grammar_issues_list, grammar_issues_count = get_grammer_correction_list(essay)
            
            spelling_errors_html = get_spelling_error_html(spelling_errors)
            grammar_issues_html = get_grammar_issues_html(grammar_issues_list)
            #print (essay_file.name + "\n Grade: " + str(essay_grade) + "\n Text: " + str(essay) + "\n\n")
            
            graded_essay = GradedEssay(essay, essay_grade, essay_file.name, confidence_score, word_count, stop_word_count, \
                spelling_error_count, spelling_errors_html, grammar_issues_html, grammar_issues_count)

            if (save_to_db):
                save_essay_to_db(essay, essay_grade, essay_title, request)

            essays.append(graded_essay)

        except Exception as e:
            print(e)
            traceback.print_exc()
            essay_grade = "None"
            graded_essay = GradedEssay(essay, essay_grade, essay_file.name, "", 0, 0, 0, "", "", 0)
            essays.append(graded_essay)

    return essays

def handle_uploaded_essay(essay_file, essay_title, save_to_db,request):

    essay = ""
    essay_grade = ""
    word_count = 0
    confidence_score = 0
    spelling_error_count = 0
    stop_word_count = 0
    spelling_errors = []
    grammar_issues_list = []
    grammar_issues_count = 0
    encoded_image = None

    for chunk in essay_file.chunks():
        essay += chunk.decode("utf-8")
    try:
        predictor = predictGrades()
        essay_grade, confidence_score = predictor.predict(essay)
        confidence_score = int((round(confidence_score,2) * 100)) #Percentage
        word_count = get_word_count(essay)
        stop_word_count = get_stop_word_count(essay)
        spelling_error_count, spelling_errors = get_spelling_error_count(essay)
        grammar_issues_list, grammar_issues_count = get_grammer_correction_list(essay)
        encoded_image = get_wordcloud_as_encoded_image(essay)
        #print("Essay Grade: " + str(essay_grade) + " %")

        if (save_to_db):
            save_essay_to_db(essay, essay_grade, essay_title, request)

    except Exception as e:
        print(e)
        traceback.print_exc()
        essay_grade = "Couldn't be determined.."

    return essay_grade,essay,word_count,stop_word_count,spelling_error_count, spelling_errors, grammar_issues_list, grammar_issues_count, confidence_score, encoded_image

def get_grammer_correction_list(essay):

    tool = language_check.LanguageTool('en-US')

    matches = tool.check(essay)

    grammar_issues = []

    for match in matches:
        if match.ruleId != 'MORFOLOGIK_RULE_EN_US': #Ignore spelling errors because another function is handling that
            grammar_issues.append(str(match))

    return grammar_issues, len(grammar_issues)

def get_word_count(essay):
    return len(essay.split(" "))

def get_stop_word_count(essay):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(essay.lower())
    stop_word_count = len([w for w in word_tokens if w in stop_words])
    #print("stop words: " + str(stop_words))
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

    if 'user' in request.session:
        user_name = request.session['user']['email']
        if request.method == 'POST':
            form = UploadEssayForm(request.POST,request.FILES)
            if form.is_valid():
                essay_title = form.cleaned_data["title"]
                save_to_db = form.cleaned_data["save_essay_checkbox"]
                essay_grade,essay,word_count,stop_word_count,spelling_error_count, spelling_errors, grammar_issues_list, grammar_issues_count, confidence_score, encoded_image = handle_uploaded_essay(request.FILES['file'], essay_title, save_to_db,request)
                #for issue in grammar_issues_list:
                #    print(str(issue))
                #encoded_image = encoded_image.decode('utf8')
                return render_to_populated_response('upload_essay.html',\
                {'title':"Auto Essay Grader",\
                'user_name': user_name,\
                'form': form,\
                'essay_grade':essay_grade,\
                'essay':essay,\
                'word_count':word_count,\
                'spelling_error_count':spelling_error_count,\
                'stop_word_count':stop_word_count,\
                'spelling_errors':spelling_errors,\
                'grammar_issues_list':grammar_issues_list,\
                'grammar_issues_count':grammar_issues_count,\
                'confidence_score':confidence_score,\
                'encoded_wordcloud':encoded_image},request)
            else:
                #This will return the same form but with errors displayed to the user
                return render_to_populated_response('upload_essay.html',\
                {'title':"Auto Essay Grader",\
                'user_name': user_name,\
                'form': form},request)

        #If its not a POST request then mimic the upload essay page
        return upload_essay(request)
    else: #Not Logged in
        return not_logged_in_redirect(request)

def submit_essay_batch(request):
    if 'user' in request.session:
        user_name = request.session['user']['email']
        if request.method == 'POST':
            form = UploadEssayFolderForm(request.POST,request.FILES)
            if form.is_valid():
                essay_title = form.cleaned_data["title"]
                save_to_db = form.cleaned_data["save_essay_checkbox"]
                #all_files = form.all_files
                valid_files = form.valid_files #Uploaded files validated within form class itself
                graded_essay_list = handle_uploaded_folder(valid_files, essay_title, save_to_db, request)
                return render_to_populated_response('upload_essay_batch.html',\
                {'title':"Auto Essay Grader",\
                'user_name': user_name,\
                'form': form,\
                'graded_essay_list':graded_essay_list,\
                'graded_essay_count':len(graded_essay_list)},request)
            else:
                return render_to_populated_response('upload_essay_batch.html',\
                {'title':"Auto Essay Grader",\
                'user_name': user_name,\
                'form': form},request)
        #If its not a POST request then mimic batch_process_essay url and display form page
        return batch_process_essays(request)
    else: #Not logged in
        return not_logged_in_redirect(request)

def get_wordcloud_as_encoded_image(essay_text):
    
    
    try:
        #wordcloud = WordCloud().generate(essay_text)

        # Display the generated image:
        # the matplotlib way:
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.show()
        #title = 'Essay Word Cloud'

        format = "PNG"

        #sio = cStringIO.StringIO()
        
        bio = BytesIO()

        wordcloud = WordCloud(background_color="white",height=600,width=600).generate(essay_text)
        image = wordcloud.to_image()
        #image.show()

        old_size = image.size
        new_size = (610, 610)
        new_im = Image.new("RGB", new_size)
        new_im.paste(image, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))

        new_im.save(bio, format)
        #plt.savefig(bio,format=format)

        #encoded_image = bio.getvalue().encode("base64").strip()

        #encoded_image = base64.b64encode(bio.getvalue()).decode('utf-8').replace('\n', '')
        #bio.close()
        #encoded_image = base64.urlsafe_b64encode(bio.getvalue())

        #plt.gcf().clear()
        encoded_image = base64.b64encode(bio.getvalue())

        return encoded_image

    except Exception as e:
        print (e)
        return None

def batch_process_essays(request):

    if 'user' in request.session:
        user_name = request.session['user']['email']
        form = UploadEssayFolderForm()
        return render_to_populated_response('upload_essay_batch.html',\
            {'title':"Auto Essay Grader",\
            'user_name': user_name,\
            'form': form},request)
    else: #Not Logged In
        return not_logged_in_redirect(request)

#Required keys: 'title' and 'user_name' for template.html that is used by all pages after log in/sign up
def upload_essay(request):

    if 'user' in request.session:
        user_name = request.session['user']['email']
        form = UploadEssayForm()
        return render_to_populated_response('upload_essay.html',\
            {'title':"Auto Essay Grader",\
            'user_name': user_name,\
            'form': form},request)
    else: #Not Logged In
        return not_logged_in_redirect(request)
