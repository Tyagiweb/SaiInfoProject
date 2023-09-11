from django.conf import settings
from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from app1 import views

urlpatterns = [
    path('',views.index,name='index'),
    path('upload/',views.upload,name='upload'),
    path('all_data/',views.all_data,name='all_data'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    