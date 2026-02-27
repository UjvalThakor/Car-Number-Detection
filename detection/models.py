from django.db import models

class VideoUpload(models.Model):
    STATUS_CHOICE = [
        ('pending','Pending'),
        ('processing','Processing'),
        ('completed','Completed'),
        ('failed','Failed'),
    ]

    video_file = models.FileField(upload_to='upload/videos/')
    original_name = models.CharField(max_length=255)
    status = models.CharField(max_length=20,choices=STATUS_CHOICE,default='pending')
    upload_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True,blank=True)
    error_message = models.TextField(blank=True,null=True)

    class Meta:
        ordering = ['-upload_at']

    def __str__(self):
        return f"Video:{self.original_name} [{self.status}]"

class DetectedPlate(models.Model):
    video = models.ForeignKey(
        VideoUpload,
        on_delete=models.CASCADE,
        related_name='detected_plates'
    )

    plate_text = models.CharField(max_length=20,db_index=True)
    confidence = models.FloatField()
    frame_number = models.IntegerField()
    bbox_x = models.IntegerField(null=True,blank=True)
    bbox_y = models.IntegerField(null=True,blank=True)
    bbox_w = models.IntegerField(null=True,blank=True)
    bbox_h = models.IntegerField(null=True,blank=True)
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-confidence']

    def __str__(self):
        return f"Plate: {self.plate_text} (conf:{self.confidence:.2f})"

    def to_dict(self):
        return {
            "plate_text": self.plate_text,
            "confidence": round(self.confidence,4),
            "frame_number": self.frame_number,
            "bounding_box":{
                "x": self.bbox_x,
                "y": self.bbox_y,
                "w": self.bbox_w,
                "h": self.bbox_h,
            }
        }