from django.urls import path
from .views import (
    DetectPlatesView,
    DetectionResultsListView,
    DetectionResultDetailView,
    AllPLatesView,
)

urlpatterns = [
    path('detect/',DetectPlatesView.as_view(),name='detect-plates'),
    path('results/',DetectionResultsListView.as_view(),name='result-view'),
    path('results/<int:pk>/',DetectionResultDetailView.as_view(),name='result-detail'),
    path('plates/',AllPLatesView.as_view(),name='all-plates'),
]