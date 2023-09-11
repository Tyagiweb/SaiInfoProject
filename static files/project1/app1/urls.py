from django.contrib import admin
from django.urls import path,include
from app1 import views

urlpatterns = [
    path('', views.index,name="index"),
    path('login/', views.login,name="login"),
    path('register/', views.register,name="register"),
    path('login/register/', views.register,name="register"),
    path('registerdata/', views.register_data,name="registerdata"),
]