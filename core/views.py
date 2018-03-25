from django.shortcuts import render

# Create your views here.

class Home(TemplateView):
    template_name = "core/spamsubmission.html"
