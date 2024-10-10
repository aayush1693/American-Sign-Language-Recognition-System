from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup/', views.signup, name='signup'),
    path('upload/', views.upload_video, name='upload_video'),
    path('result/<int:video_id>/', views.result, name='result'),
]