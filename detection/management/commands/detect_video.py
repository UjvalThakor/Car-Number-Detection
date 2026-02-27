import sys
import os
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Detect number plates from a video file'

    def add_arguments(self, parser):
        parser.add_argument('video_path', type=str)

    def handle(self, *args, **kwargs):
        from detection.plate_dectector import NumberPlateDetector

        video_path = kwargs['video_path']

        if not os.path.exists(video_path):
            self.stdout.write(self.style.ERROR(f'❌ File not found: {video_path}'))
            return

        self.stdout.write('🚗 Starting detection...')

        detector = NumberPlateDetector(
            confidence_threshold=0.3,
            gpu=False,
        )

        results = detector.process_video(video_path)

        self.stdout.write('=' * 50)
        self.stdout.write(f' Total Plates Found: {len(results)}')
        self.stdout.write('=' * 50)

        plates_list = [r['plate_text'] for r in results]
        self.stdout.write(f' Plates List: {plates_list}')

        for i, plate in enumerate(results, 1):
            self.stdout.write(
                f"{i}. Plate: {plate['plate_text']} | "
                f"Confidence: {plate['confidence']:.0%} | "
                f"Frame: {plate['frame_number']}"
            )