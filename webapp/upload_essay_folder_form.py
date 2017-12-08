from django import forms
from django.core.exceptions import ValidationError
import os

MAX_ESSAYS = 20

class UploadEssayFolderForm(forms.Form):
    title = forms.CharField(label="Essay Prompt: ",max_length=50, required=True,widget=forms.TextInput(attrs={'class':'form-control','placeholder':'Essay Prompt'}))

    save_essay_checkbox = forms.BooleanField(label="Save Essays To DB: ",initial=True,required=False)

    files = forms.FileField(label="Upload a folder containing .txt files. All unique txt files within the folder and sub-folder(s) will be graded.",required=True,widget=forms.ClearableFileInput(attrs=
        {'multiple': True, 'webkitdirectory': True, 'directory': True}))

    def clean(self):

        self.all_files = self.files.getlist('files')
        self.valid_files = self.get_valid_files(self.all_files)
        if len(self.valid_files) == 0:
            raise ValidationError('No txt files found! Please upload a folder that contains atleast 1 txt file to grade.')
        elif len(self.valid_files) > MAX_ESSAYS:
            error_text = "A maximum of " + str(MAX_ESSAYS) + " essays can be graded. You have uploaded " + str(len(self.valid_files)) + " essays instead. Please try again."
            raise ValidationError(error_text)

    def get_valid_files(self, all_files):
        valid_files = []
        valid_file_names = []
        for essay_file in all_files:
            ext = os.path.splitext(essay_file.name)[1]
            if ext.lower() == ".txt" and essay_file.name not in valid_file_names:
                valid_files.append(essay_file)
                valid_file_names.append(essay_file.name)
        return valid_files