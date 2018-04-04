from django.shortcuts import render
from django.views.generic import FormView
# Create your views here.

from .forms import *
from .models import *


def Home(request):
	form = ClassifierForm()
	if request.method == 'POST':
		form = ClassifierForm(request.POST)
		if form.is_valid():
			timestamp = datetime.now()
			print(timestamp)
			form.times = timestamp
			form.save()

	else:
		form = ClassifierForm()

	return render(request,'core/spamsubmission.html',{'form':form })    
