from django.db import models
import datetime

# Create your models here.
class Classifer(models.Model):
    inputtext = models.TextField()
    times = models.DateField()

    def __str__(self):
        return self.input
