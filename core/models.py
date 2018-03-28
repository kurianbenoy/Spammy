from django.db import models
import datetime

# Create your models here.
class Classifer(models.Model):
    input = models.TextField()
    times = models.DateField()

    def date_status(self):
        if self.times < datetime.date(2018,4,4):
            return "Testing model"
        if self.times > datetime.date(2018,4,4):
            return "Training data"

    def __str__(self):
        return self.name
