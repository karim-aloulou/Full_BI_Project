from django import forms
from django.core import validators

class predictForm(forms.Form):
    FK_Customer = forms.CharField(validators=[validators.RegexValidator(r'^\d+$', 'Enter a valid ID')])
    