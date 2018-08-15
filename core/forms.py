from django.forms import ModelForm, Textarea

try:
	from .models import Classifer
except:
	from models import Classifier

class ClassifierForm(ModelForm):	

    class Meta:
        model = Classifer
        fields = ['inputtext',]
        widget = {
            'inputtext': Textarea(attrs={'cols': 1000, 'rows': 500}),
        }
    

    # def __init__(self,*args,**kwargs):
    # 	super().__init__(*args,**kwargs)
    # 	self.fields['inputtext'].widget.attrs={'class':'spamform'}    
