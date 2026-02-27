from django.shortcuts import render
import os
import logging
from datetime import datetime

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .models import VideoUpload, DetectedPlate
from .serializers import (
    VideoUploadSerializer,
    VideoUploadInputSerializers,
    DetectedPlateSerializer,
)

from .plate_dectector import NumberPlateDetector

logger = logging.getLogger(__name__)


def save_plates_to_db(video_obj: VideoUpload, results: list) -> None:
    plate_objects = []
    for r in results:
        bbox = r.get("bounding_box") or (None, None, None, None)
        plate_objects.append(DetectedPlate(
            video=video_obj,
            plate_text=r["plate_text"],
            confidence=r["confidence"],
            frame_number=r["frame_number"],
            bbox_x=bbox[0] if bbox else None,
            bbox_y=bbox[1] if bbox else None,
            bbox_w=bbox[2] if bbox else None,
            bbox_h=bbox[3] if bbox else None,
        ))
    DetectedPlate.objects.bulk_create(plate_objects)


class DetectPlatesView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        input_serializer = VideoUploadInputSerializers(data=request.data)
        if not input_serializer.is_valid():
            return Response(
                {"success": False, "errors": input_serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        video_file = input_serializer.validated_data["video_file"]

        video_obj = VideoUpload.objects.create(
            video_file=video_file,
            original_name=video_file.name,
            status="processing",
        )

        video_path = os.path.join(settings.MEDIA_ROOT, str(video_obj.video_file))

        try:
            detector = NumberPlateDetector(
                confidence_threshold=0.3,
                gpu=False,
            )

            results = detector.process_video(video_path)
            save_plates_to_db(video_obj, results)

            video_obj.status = "completed"
            video_obj.processed_at = datetime.now()
            video_obj.save()

            plates_list = [r["plate_text"] for r in results]

            return Response(
                {
                    "success": True,
                    "video_id": video_obj.id,
                    "total_plate_found": len(plates_list),
                    "plate_list": plates_list,
                    "detailed_results": results,
                    "message": "Detection complete",
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            video_obj.status = "failed"
            video_obj.error_message = str(e)
            video_obj.save()
            logger.error(f"Detection Failed for video {video_obj.id}: {e}")

            return Response(
                {"success": False, "error": f"Detection Failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DetectionResultsListView(APIView):
    def get(self, request):
        videos = VideoUpload.objects.prefetch_related('detected_plates').all()
        serializer = VideoUploadSerializer(videos, many=True)

        return Response(
            {
                "success": True,
                "total_videos": videos.count(),
                "results": serializer.data,
            },
            status=status.HTTP_200_OK
        )


class DetectionResultDetailView(APIView):
    def get_object(self, pk):
        try:
            return VideoUpload.objects.prefetch_related('detected_plates').get(pk=pk)
        except VideoUpload.DoesNotExist:
            return None

    def get(self, request, pk):
        video_obj = self.get_object(pk)
        if not video_obj:
            return Response(
                {"success": False, "error": "Video not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = VideoUploadSerializer(video_obj)
        plates_list = list(video_obj.detected_plates.values_list('plate_text', flat=True))

        return Response(
            {
                "success": True,
                "plates_list": plates_list,
                "detail": serializer.data,
            },
            status=status.HTTP_200_OK
        )

    def delete(self, request, pk):
        video_obj = self.get_object(pk)
        if not video_obj:
            return Response(
                {"success": False, "error": "Video not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        if video_obj.video_file and os.path.isfile(video_obj.video_file.path):
            os.remove(video_obj.video_file.path)

        video_obj.delete()
        return Response(
            {"success": True, "message": "Record Deleted Successfully"},
            status=status.HTTP_200_OK
        )


class AllPLatesView(APIView):
    def get(self, request):
        plates = DetectedPlate.objects.all()

        min_conf = request.query_params.get('min_confidence')
        search = request.query_params.get('search', '').upper()

        if min_conf:
            try:
                plates = plates.filter(confidence__gte=float(min_conf))
            except ValueError:
                pass

        if search:
            plates = plates.filter(plate_text__icontains=search)

        plates_list = list(plates.values_list("plate_text", flat=True).distinct())
        serializer = DetectedPlateSerializer(plates, many=True)

        return Response(
            {
                "success": True,
                "total_plates": len(plates_list),
                "plates_list": plates_list,
                "plates_detail": serializer.data,
            },
            status=status.HTTP_200_OK
        )