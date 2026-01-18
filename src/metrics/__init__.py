"""
Metrics Collection Module

Includes:
- NASA-TLX questionnaire
- Task Completion metrics
- Cognitive Load Calculator (CLI) - composite scoring using gaze, emotion, and mouse metrics
"""

from src.metrics.cognitive_load_metrics import (
    TaskMetrics,
    NASATLX,
    MetricsCollector
)

from src.metrics.cognitive_load_calculator import (
    CognitiveLoadCalculator,
    GazeMetrics,
    EmotionMetrics,
    MouseMetrics
)

__all__ = [
    'TaskMetrics',
    'NASATLX',
    'MetricsCollector',
    'CognitiveLoadCalculator',
    'GazeMetrics',
    'EmotionMetrics',
    'MouseMetrics'
]