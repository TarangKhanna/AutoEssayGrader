from django import forms
from django.core.exceptions import ValidationError
import os

class UploadLoginForm(forms.Form):
    title = forms.CharField(label="Essay Title: ",max_length=50, required=True,widget=forms.TextInput(attrs={'class':'form-control','placeholder':'Essay Title'}))

    file = forms.FileField(label="Upload a .txt file:",required=True)

