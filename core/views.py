from django.shortcuts import render
#from django.views.generic import TemplateView
from django.views.generic import FormView
# Create your views here.

class Home(FormView):
    form_class =  ClassifierForm
    template_name = "core/spamsubmission.html"
    
