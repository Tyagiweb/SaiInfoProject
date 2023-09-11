from django.contrib import admin
from django.urls import path,include
from app1 import views

urlpatterns = [
    path('', views.register,name='register'),
    path('data/register', views.register,name='register'),
    path('login/', views.login,name='login'),
    path('data/login/', views.login,name='login'),
    path('registerform/', views.registerform,name='registerform'),
    path('data/', views.data,name='data'),
    path('viewdata/', views.viewdata,name='viewdata'),
    path('login/viewdata2/', views.viewdata,name='viewdata2'),
    path('login/validuser', views.validuser,name='validuser'),
    

]