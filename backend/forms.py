from django import forms
from db.models import Video

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video_file']