from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from .forms import VideoUploadForm
from db.models import Video, Result
from models.transformer_model import process_video as process_video_model

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

@login_required
def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save(commit=False)
            video.user = request.user
            video.save()
            process_video(video)
            return redirect('result', video_id=video.id)
    else:
        form = VideoUploadForm()
    return render(request, 'upload_video.html', {'form': form})

@login_required
def result(request, video_id):
    result = Result.objects.get(video__id=video_id)
    return render(request, 'result.html', {'result': result})

def process_video(video):
    video.status = 'processing'
    video.save()

    recognized_text = process_video_model(video.video_file.path)

    Result.objects.create(video=video, recognized_text=recognized_text)

    video.status = 'completed'
    video.save()