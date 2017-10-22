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

# Get a reference to the auth service
auth = firebase.auth()