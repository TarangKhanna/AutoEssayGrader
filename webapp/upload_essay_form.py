from django import forms
from django.core.exceptions import ValidationError
import os

class UploadEssayForm(forms.Form):
    title = forms.CharField(label="Essay Title: ",max_length=50, required=True,widget=forms.TextInput(attrs={'class':'form-control','placeholder':'Essay Title'}))

    save_essay_checkbox = forms.BooleanField(label="Save Essay To DB: ",initial=True,required=False)

    file = forms.FileField(label="Upload a .txt file:",required=True)

    def clean(self):

    	if ".txt" not in os.path.splitext(self.cleaned_data['file'].name)[1]:
            raise ValidationError('Only .txt files are accepted !')
