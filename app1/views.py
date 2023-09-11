

# Create your views here.

from django.shortcuts import render
from .models import Student

# Create your views here.

def index(request):
    return render(request,'index.html')

def login(request):
    return render(request,'login.html')

def register(request):
    return render(request,'register.html')


def register_data(request):
    firstname=request.POST['firstname']
    middlename=request.POST['middlename']
    lastname=request.POST['lastname']
    phone=request.POST['phone']
    adress=request.POST['adress']
    email=request.POST['email']
    psw=request.POST['psw']
    user=Student.objects.create(Firstname=firstname,Middlename=middlename,Lastname=lastname,mobile=phone,
                                Adress=adress,Email=email,Psw=psw)
    return render(request,'submit.html')
