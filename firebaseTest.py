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
#	/essays
#		/essayid
#			/grade
#			/confidence

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

config = {
  "apiKey": "AIzaSyBOoiObBt9iWJg04Kog-qBqsXwYIjiKkwk",
  "authDomain": "essaygrader-93d7d.firebaseapp.com",
  "databaseURL": "https://essaygrader-93d7d.firebaseio.com",
  "storageBucket": "essaygrader-93d7d.appspot.com",
}

firebase = pyrebase.initialize_app(config)

# Get a reference to the database service
db = firebase.database()

email = "shivanktibrewal@gmail.com"
password = "hello123"

# Get a reference to the auth service
auth = firebase.auth()

#auth.create_user_with_email_and_password(email, password)

user = auth.sign_in_with_email_and_password(email, password)

print(user)


