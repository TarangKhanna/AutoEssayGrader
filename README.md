# AutoEssayGrader
<br />
Utilizes machine learning to grade essays<br />
<br />
Requirements:<br />
Need to have Python 3<br />
<br />
Install Django:<br />
pip install Django<br />
<br />
Install PyreBase:<br />
pip install pyrebase<br />
More info on pyrebase: https://github.com/thisbejim/Pyrebase#authentication<br />
<br />
Install Firebase:<br />
sudo pip install python-firebase<br />
<br />
Install language_check:<br />
pip install language_check<br />
<br />
Install Copyleaks Python SDK:<br />
pip3 install copyleaks<br />
<br />
Migrate Changes in modules:<br />
python manage.py migrate<br />
<br />
Run server:<br />
python manage.py runserver<br />
<br />
Navigate to:<br />
http://127.0.0.1:8000/webapp/upload_essay/<br />
<br />
<br />
<br />
SIGN IN DETAILS:<br />
Email: shivanktibrewal@gmail.com<br />
Password: hello123<br />
<br />
<br />
<br />
essaygrader/ - contains the essay grading component of the project. Also contains the django settings.py, urls.py
<br />
<br />
webapp/ - contains all the web rendering stuff
<br />
<br />
Webapp folders and files:
<br /><br />
static/ - static files like bootstrap etc<br />
templates/ - html files that are to be rendered. All files extend template.html (except the login and sign up pages)<br />
views.py - main rendering component with methods that handle each URL<br />
urls.py - contains the valid urls and the methods that are called in views.py<br />
request_handler.py - utility file that aids in rendering html files (render_to_populated_response) and getting user inputs (get_param_with_default).<br />

