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

# Get a reference to the auth service
auth = firebase.auth()