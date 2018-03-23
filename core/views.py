from django.shortcuts import render
from django.views.genric import TemplateView
# Create your views here.

class Home(TemplateView):
    template_name = "core/spamsubmission.html"
