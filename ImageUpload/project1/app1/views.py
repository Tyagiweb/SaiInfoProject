from django.shortcuts import render,redirect
from .models import *

# Create your views here.

def index(request):
    return render(request,'index.html')

def upload(request):
    if request.method=='POST':
        imagename=request.POST['name']
        image2=request.FILES['image']
        user=ImageUpload.objects.create(Name=imagename,Image=image2)
        #return render(request,'submit.html')
        return redirect(all_data)
    

def all_data(request):
    user=ImageUpload.objects.all()
    #return render(request,'index.html',{'key2':user})
    return render(request,'imageseen.html',{'key2':user})
    

