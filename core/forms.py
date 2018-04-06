from django.forms import ModelForm, Textarea

from .models import Classifer


class ClassifierForm(ModelForm):	

    class Meta:
        model = Classifer
        fields = ['inputtext',]
        widget = {
            'inputtext': Textarea(attrs={'cols': 20, 'rows': 100}),
        }
    

    # def __init__(self,*args,**kwargs):
    # 	super().__init__(*args,**kwargs)
    # 	self.fields['inputtext'].widget.attrs={'class':'spamform'}    
