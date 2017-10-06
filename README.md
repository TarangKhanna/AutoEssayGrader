# AutoEssayGrader

Utilizes machine learning to grade essays

Requirements:
Need to have Python 3

Install Django:
pip install Django

Install Firebase:
sudo pip install python-firebase`

Migrate Changes in modules:
python manage.py migrate

Run server:
python manage.py runserver

Navigate to:
http://127.0.0.1:8000/webapp/upload_essay/

essaygrader/ - contains the essay grading component of the project. Also contains the django settings.py, urls.py

webapp/ - contains all the web rendering stuff

Webapp folders and files:

static/ - static files like bootstrap etc
templates/ - html files that are to be rendered. All files extend template.html (except the login and sign up pages)
views.py - main rendering component with methods that handle each URL
urls.py - contains the valid urls and the methods that are called in views.py
request_handler.py - utility file that aids in rendering html files (render_to_populated_response) and getting user inputs (get_param_with_default).

