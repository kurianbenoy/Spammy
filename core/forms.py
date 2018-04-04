from django.forms import ModelForm

from .models import Classifer


class ClassifierForm(ModelForm):
    class Meta:
        model = Classifer
        fields = ['inputtext',]
        
