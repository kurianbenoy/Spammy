import datetime
from django.shortcuts import render
from django.views.generic import FormView
# Create your views here.

from .forms import ClassifierForm
from .models import Classifer


def Home(request):
	form = ClassifierForm()
	if request.method == 'POST':
		form = ClassifierForm(request.POST)
		if form.is_valid():
			timestamp = datetime.datetime.now()
			print(timestamp)
			form = ClassifierForm(request.POST)
			author = form.save(commit=False)
			author.times = timestamp
			author.save()
		else :
			print(form.errors)

	else:
		form = ClassifierForm()

	return render(request,'core/spamsubmission.html',{'form':form })    
