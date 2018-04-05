from django.db import models
import datetime

# Create your models here.
class Classifer(models.Model):
    inputtext = models.TextField()
    times = models.DateTimeField(null=True)

    def __str__(self):
        return self.inputtext
