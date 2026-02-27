from rest_framework import serializers
from .models import VideoUpload,DetectedPlate

class DetectedPlateSerializer(serializers.ModelSerializer):

    class Meta:
        model = DetectedPlate
        fields = ['plate_text','confidence','frame_number',
                  'bbox_x','bbox_y','bbox_w','bbox_h','detected_at']

class VideoUploadSerializer(serializers.ModelSerializer):

    detected_plates = DetectedPlateSerializer(many=True,read_only=True)
    plates_list = serializers.SerializerMethodField()

    class Meta:
        model = VideoUpload
        fields = ['id','original_name','status','upload_at',
                  'processed_at','plates_list','detected_plates','error_message']

    def get_plates_list(self,obj):
        return list(obj.detected_plates.values_list('plate_text',flat=True))

class VideoUploadInputSerializers(serializers.Serializer):
    video_file =serializers.FileField()

    ALLOWED_EXTENSIONS = ['.mp4','.avi','.mov','.mkv','.webm']
    MAX_FILE_SIZE_MB = 500

    def validate_video_file(self,value):
        import os

        ext = os.path.splitext(value.name)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise serializers.ValidationError(
                f"Unsupported format '{ext}'.Allowed:{self.ALLOWED_EXTENSIONS}"
            )

        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        if value.size > max_bytes:
            raise serializers.ValidationError(
                f"File too large. Max allowed:{self.MAX_FILE_SIZE_MB}MB"
            )
        return value