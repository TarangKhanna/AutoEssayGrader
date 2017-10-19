from django import forms
from django.core.exceptions import ValidationError
import os

class UploadLoginForm(forms.Form):
    username = forms.CharField(label="Email: ",max_length=50, required=True,widget=forms.TextInput(attrs={'class':'form-control','placeholder':'Username'}))
    password = forms.CharField(label="Password: ",max_length=50, required=True,widget=forms.PasswordInput(attrs={'class':'form-control','placeholder':'Password'}))

