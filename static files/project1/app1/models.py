from django.db import models

# Create your models here.
class Student(models.Model):
    Firstname=models.CharField(max_length=10)
    Middlename=models.CharField(max_length=10)
    Lastname=models.CharField(max_length=10)
    Mobile=models.CharField(max_length=10)
    Adress=models.CharField(max_length=20)
    Email=models.CharField(max_length=20)
    Psw=models.CharField(max_length=20)