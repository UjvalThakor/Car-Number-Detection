from django.contrib import admin
from .models import VideoUpload,DetectedPlate

@admin.register(VideoUpload)
class VideoUploadAdmin(admin.ModelAdmin):
    list_display = ['id','original_name','status','upload_at','processed_at']
    list_filter = ['status']
    search_fields = ['original_name']
    readonly_fields = ['upload_at','processed_at']

@admin.register(DetectedPlate)
class DetectedPlateAdmin(admin.ModelAdmin):
    list_display = ['id','plate_text','confidence','frame_number','video','detected_at']
    list_filter = ['detected_at']
    search_fields = ['plate_text']
    readonly_fields = ['detected_at']