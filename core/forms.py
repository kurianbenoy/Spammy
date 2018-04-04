from django.forms import ModelForm

from .models import Classifer


class ClassifierForm(ModelForm):	

    class Meta:
        model = Classifer
        fields = ['inputtext',]
    

    def __init__(self,*args,**kwargs):
    	super().__init__(*args,**kwargs)
    	self.fields['inputtext'].widget.attrs={'class':'spamform'}    
