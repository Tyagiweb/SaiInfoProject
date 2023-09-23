from django.http import HttpResponseRedirect
from django.shortcuts import redirect, render
from .models import Student

# Create your views here.
def register(request):
    return render(request,'home.html')

def login(request):
    return render(request,'login.html')

def registerform(request):
    return render(request,'register.html')


def data(request):
    firstname=request.POST['firstname']
    middlename=request.POST['middlename']
    lastname=request.POST['lastname']
    phone=request.POST['phone']
    adress=request.POST['adress']
    email=request.POST['email']
    psw=request.POST['psw']
    user=Student.objects.create(Firstname=firstname,Middlename=middlename,Lastname=lastname,Mobile=phone,
                                Adress=adress,Email=email,Psw=psw)
    return redirect('login')                            
    
    

def viewdata(request):
    all_data=Student.objects.all()
    return render(request,'submit.html',{'key1':all_data})


    
def validuser(request):
    if request.method=="POST":
        firstname=request.POST['username']
        password=request.POST['password']

        user=Student.objects.get(Firstname=firstname)
        if user:
            if user.Psw==password:
                request.session['Name']=user.Firstname
                request.session['Email']=user.Email
                return render(request,'Afterlogin.html')
            else:
                message="Password does not match"
                return render(request,'login.html',{'msg':message})
        else:
            message="User not exist"
            return render(request,'register.html',{'msg':message})   


#Edit
def EditPage(request,pk):
    get_data=Student.objects.get(id=pk) 
    return render(request,'edit.html',{'key2':get_data})  


#Update

def UpdateData(request,pk):
    udata=Student.objects.get(id=pk)
    udata.Firstname=request.POST['firstname']
    udata.Middlename=request.POST['middlename']
    udata.Lastname=request.POST['lastname']
    udata.Mobile=request.POST['phone']
    udata.Adress=request.POST['adress']
    udata.Email=request.POST['email']
    udata.Psw=request.POST['psw']
    udata.save()
    return redirect('login')

    
      





