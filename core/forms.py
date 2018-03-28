from django.forms import ModelForm
from .models import Classifer


class ClassiferForm(ModelForm):
    class Meta:
        model = Classifer
        fields = [input]
        
