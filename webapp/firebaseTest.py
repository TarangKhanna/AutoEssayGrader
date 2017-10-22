#pip install requests==1.1.0
#pip install python-firebase
"""
from firebase import firebase
import json
import datetime
import random
firebase = firebase.FirebaseApplication('https://essaygrader-93d7d.firebaseio.com', None)

#Read from database 

#get users table from database
result = firebase.get('/users', None)
print (result)
#get user1
result = firebase.get('/users', "user1")
print (result)
#get all essays for all users
#/users
#  /userid
# /essays
#   /essayid
#     /grade
#     /confidence

result = firebase.get('/essays', None)
print (result)

#write to database

#data = {'first_name': 'j', 'last_name' :'a', 'email':'a@a.com'}
#sent = json.dumps(data)
#result = firebase.post('/users', sent)
#print result
data = {'name': 'Ozgur Vatansever', 'email': 'ov@gmail.com',
            'username': 'ovat', 'password':"test"}
x = random.randint(1,1000000)
snapshot = firebase.patch('/users/'+str(x), data)
print(snapshot['name'])
"""

import pyrebase

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

path_to_service_account = ROOT_DIR + "/" + "essaygrader-93d7d-firebase-adminsdk-qczsl-9482a69ba0.json"

config = {
  "apiKey": "AIzaSyBOoiObBt9iWJg04Kog-qBqsXwYIjiKkwk",
  "authDomain": "essaygrader-93d7d.firebaseapp.com",
  "databaseURL": "https://essaygrader-93d7d.firebaseio.com",
  "storageBucket": "essaygrader-93d7d.appspot.com",
  "serviceAccount": path_to_service_account
}

firebase = pyrebase.initialize_app(config)

# Get a reference to the database service
db = firebase.database()

email = "shivanktibrewal@gmail.com"
password = "hello123"

# Get a reference to the auth service
auth = firebase.auth()

#SIGN UP
#auth.create_user_with_email_and_password(email, password)

#SIGN IN
user = auth.sign_in_with_email_and_password(email, password)

user_id_token = user['idToken']

print(user)
"""
user should be something like this:
{'idToken': 'eyJhbGciOiJSUzI1NiIsImtpZCI6ImZjNmRlYzAwM2RiZmQzZGMyMmFjY2JiYTQ4ZmZhNmZkYjNlNzU2NWIifQ.eyJpc3MiOiJodHRwczovL3NlY3VyZXRva2VuLmdvb2dsZS5jb20vZXNzYXlncmFkZXItOTNkN2QiLCJhdWQiOiJlc3NheWdyYWRlci05M2Q3ZCIsImF1dGhfdGltZSI6MTUwODcwMzc3NCwidXNlcl9pZCI6Ikh4TEk2YU5IcGJlOWh4Nlpob0Vndk9yTU02QzMiLCJzdWIiOiJIeExJNmFOSHBiZTloeDZaaG9FZ3ZPck1NNkMzIiwiaWF0IjoxNTA4NzAzNzc0LCJleHAiOjE1MDg3MDczNzQsImVtYWlsIjoic2hpdmFua3RpYnJld2FsQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6eyJlbWFpbCI6WyJzaGl2YW5rdGlicmV3YWxAZ21haWwuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoicGFzc3dvcmQifX0.BIEdTk6ZBMKMZGKDwJyaf1scC-uRQQ0wsPhQ2tyGo_mBYjrvY05jYt-LvLHkG93ER7iyaYQvyIaFThZImpg_xIj9NJX6FAM_fOrICwHmoMCBx0LVBl_5_Jn7_FfxJtPjMVZaHs6ZeUu1F3-QWaYVasfQun13TNoDIwaFYmeLrsuxOFxcfvxHwE7g_3JVHyTov-bOv3OvHhNl8ICDcd5YXohLnxKe1-9_HaJWYasexum1Ta2N5PLltZ0i2LtsqFdzag5vEVtfYvc7bd_4k9wELpy1P7TOn39if337mzkDtoNypJyIOM6lrwz3II3wy_urbXWxbRsyswR1sZvcrMXbNw', 'displayName': '', 'kind': 'identitytoolkit#VerifyPasswordResponse', 'refreshToken': 'APWA_kryFvVnTkn8ktAyjbyIrsEX0Lvg-EY5DiO2yt2Gj5SwkQ5JmXgfXDAjGGHknXOpjYxLXp9JpRdJVBKdUSJVF2RBYosw3C2GViyRh033f6Di3lmoAFVlMz4VpgD3S_pxMs54xDGzP1fJ9bQXopV32X0Sr_dxXH-hklN-u1jBAgezB_Ig9C-w2PQ9D6CF-fxqM4ZhWbImdgEWaNgdD33J0Z9MhGnU3fhUeLuhSe7eAwXY1gN67tY', 'localId': 'HxLI6aNHpbe9hx6ZhoEgvOrMM6C3', 'registered': True, 'email': 'shivanktibrewal@gmail.com', 'expiresIn': '3600'}
"""

###SAVING DATA
"""
data = {
        "email":"okarim@gmail.com",\
        "title":"hello3",\
        "essay_text":"hello essay3",\
        "essay_grade":"hello grade3"
      }

# Pass the user's idToken to the push method
results = db.child("user_essay").push(data, user_id_token)
"""

"""
Creates this tree:

-essaygrader
  -user_essay
    -random_generated_id
      *data*
"""

#Retrieving Data
all_essays = db.child("user_essay").get()

print("\n\nAll Essays: ")
for essay in all_essays.each():
  print(str("\nKey: " + essay.key()))
  print(str("Value: " + str(essay.val())))

#Key is randomly generated
#Value is something like: Value: {'essay_grade': 83.0, 'essay_text': 'A tr....

all_essays_by_email = db.child("user_essay").order_by_child("email").equal_to("shivanktibrewal@gmail.com").get()
print("\n\nAll Essays By Email: ")
for essay in all_essays_by_email.each():
  print(str("\nKey: " + essay.key()))
  print(str("Value: " + str(essay.val())))

