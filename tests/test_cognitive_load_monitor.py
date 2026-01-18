"""
Tests for cognitive load monitoring and CLI calculation
"""

import pytest
from datetime import datetime, timedelta
from src.adaptation_engine.cognitive_load_monitor import CognitiveLoadMonitor
from src.metrics.cognitive_load_calculator import (
    CognitiveLoadCalculator,
    GazeMetrics,
    EmotionMetrics,
    MouseMetrics
)


class TestCognitiveLoadMonitor:
    """Tests for the CognitiveLoadMonitor class"""
    
    def test_initialization(self):
        """Test cognitive load monitor initialization"""
        monitor = CognitiveLoadMonitor()
        assert monitor.gaze_wander_threshold == 0.3
        assert monitor.fixation_duration_threshold == 3.0
        assert len(monitor.gaze_history) == 0
        assert monitor.cli_calculator is not None

    def test_gaze_update(self):
        """Test gaze update"""
        monitor = CognitiveLoadMonitor()
        monitor.update_gaze(0.5, 0.5)
        assert len(monitor.gaze_history) == 1
        assert monitor.gaze_history[0].x == 0.5
        assert monitor.gaze_history[0].y == 0.5

    def test_gaze_wander_detection(self):
        """Test gaze wander detection"""
        monitor = CognitiveLoadMonitor(gaze_wander_threshold=0.1)
        
        # Create wandering gaze pattern
        for i in range(20):
            x = (i % 5) * 0.2
            y = (i // 5) * 0.2
            monitor.update_gaze(x, y)
        
        assert monitor.detect_gaze_wander() == True

    def test_fixation_duration_detection(self):
        """Test fixation duration detection"""
        monitor = CognitiveLoadMonitor(fixation_duration_threshold=2.0)
        
        # Simulate long fixation
        start_time = datetime.now() - timedelta(seconds=3)
        monitor.current_fixation_start = start_time
        monitor.current_fixation_location = (0.5, 0.5)
        
        is_long, duration = monitor.detect_fixation_duration()
        assert is_long is True
        assert duration > 2.0

    def test_cognitive_overload_check(self):
        """Test cognitive overload detection"""
        monitor = CognitiveLoadMonitor()
        
        # Update with high negative affect
        emotion_result = {
            'negative_affect_score': 0.7,
            'emotions': {'angry': 0.4, 'sad': 0.3, 'neutral': 0.2, 'happy': 0.1},
            'dominant_emotion': 'angry'
        }
        monitor.update_emotion(emotion_result)
        
        # Create wandering gaze
        for i in range(20):
            monitor.update_gaze(i * 0.05, i * 0.05)
        
        result = monitor.check_cognitive_overload()
        assert 'overload_detected' in result
        assert 'cognitive_load_score' in result
        assert 'cli' in result
        assert 'load_level' in result
        assert 'gaze_score' in result
        assert 'emotion_score' in result
        assert 'mouse_score' in result
        assert 'triggers' in result
        assert 'details' in result

    def test_mouse_update(self):
        """Test mouse tracking update"""
        monitor = CognitiveLoadMonitor()
        
        # Simulate mouse movement and clicks
        for i in range(10):
            monitor.update_mouse(100 + i * 10, 100 + i * 5, clicked=(i % 3 == 0))
        
        result = monitor.check_cognitive_overload()
        assert 'mouse_score' in result
        
    def test_cli_weights(self):
        """Test that CLI weights are applied correctly"""
        monitor = CognitiveLoadMonitor(
            weight_gaze=0.6,
            weight_emotion=0.3,
            weight_mouse=0.1
        )
        
        # Weights should be normalized
        calculator = monitor.cli_calculator
        total_weight = calculator.weight_gaze + calculator.weight_emotion + calculator.weight_mouse
        assert abs(total_weight - 1.0) < 0.001

    def test_reset(self):
        """Test monitor reset"""
        monitor = CognitiveLoadMonitor()
        
        # Add some data
        monitor.update_gaze(0.5, 0.5)
        monitor.update_emotion({'negative_affect_score': 0.5, 'emotions': {}, 'dominant_emotion': 'neutral'})
        monitor.check_cognitive_overload()
        
        # Reset
        monitor.reset()
        
        assert len(monitor.gaze_history) == 0
        assert monitor.cognitive_load_score == 0.0
        assert monitor.overload_detected is False


class TestCognitiveLoadCalculator:
    """Tests for the CognitiveLoadCalculator class"""
    
    def test_initialization(self):
        """Test calculator initialization"""
        calc = CognitiveLoadCalculator()
        assert calc.weight_gaze == 0.5
        assert calc.weight_emotion == 0.3
        assert calc.weight_mouse == 0.2
    
    def test_gaze_metrics_update(self):
        """Test gaze metrics calculation"""
        calc = CognitiveLoadCalculator()
        
        # Simulate gaze movements
        for i in range(20):
            x = 0.5 + (i % 5) * 0.02
            y = 0.5 + (i // 5) * 0.02
            metrics = calc.update_gaze(x, y)
        
        assert isinstance(metrics, GazeMetrics)
        assert metrics.gaze_dispersion >= 0
    
    def test_emotion_metrics_update(self):
        """Test emotion metrics calculation"""
        calc = CognitiveLoadCalculator()
        
        emotion_result = {
            'emotions': {
                'angry': 0.3,
                'sad': 0.2,
                'happy': 0.1,
                'neutral': 0.4
            },
            'dominant_emotion': 'neutral',
            'negative_affect_score': 0.5
        }
        
        metrics = calc.update_emotion(emotion_result)
        
        assert isinstance(metrics, EmotionMetrics)
        assert metrics.negative_affect_score == 0.5
        assert 0 <= metrics.valence <= 1
        assert 0 <= metrics.arousal <= 1
    
    def test_mouse_metrics_update(self):
        """Test mouse metrics calculation"""
        calc = CognitiveLoadCalculator()
        
        # Simulate mouse movements
        for i in range(20):
            clicked = i % 5 == 0
            calc.update_mouse(100 + i * 10, 200 + i * 5, clicked=clicked)
        
        metrics = calc.current_mouse
        assert isinstance(metrics, MouseMetrics)
        assert metrics.click_rate >= 0
        assert 0 <= metrics.path_efficiency <= 1
    
    def test_cli_calculation(self):
        """Test CLI composite score calculation"""
        calc = CognitiveLoadCalculator()
        
        # Add data for all components
        for i in range(10):
            calc.update_gaze(0.5 + i * 0.01, 0.5)
        
        calc.update_emotion({
            'emotions': {'neutral': 0.8, 'happy': 0.2},
            'dominant_emotion': 'neutral',
            'negative_affect_score': 0.1
        })
        
        for i in range(10):
            calc.update_mouse(100 + i * 10, 100)
        
        result = calc.calculate_cli()
        
        assert 'cli' in result
        assert 'gaze_score' in result
        assert 'emotion_score' in result
        assert 'mouse_score' in result
        assert 'load_level' in result
        assert result['load_level'] in ['low', 'medium', 'high']
        assert 0 <= result['cli'] <= 1
    
    def test_load_level_thresholds(self):
        """Test load level categorization"""
        calc = CognitiveLoadCalculator()
        
        # Low load scenario
        calc.update_gaze(0.5, 0.5)
        calc.update_emotion({
            'emotions': {'happy': 0.8, 'neutral': 0.2},
            'dominant_emotion': 'happy',
            'negative_affect_score': 0.0
        })
        
        result = calc.calculate_cli()
        # With minimal stress indicators, should be low load
        assert result['cli'] <= 0.5
    
    def test_interpretation(self):
        """Test load interpretation generation"""
        calc = CognitiveLoadCalculator()
        
        # Add some data
        for i in range(10):
            calc.update_gaze(0.5, 0.5)
        calc.update_emotion({
            'emotions': {'neutral': 1.0},
            'dominant_emotion': 'neutral',
            'negative_affect_score': 0.0
        })
        
        result = calc.calculate_cli()
        interpretation = calc.get_load_interpretation(result)
        
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
    
    def test_reset(self):
        """Test calculator reset"""
        calc = CognitiveLoadCalculator()
        
        # Add data
        calc.update_gaze(0.5, 0.5)
        calc.update_emotion({'emotions': {}, 'dominant_emotion': 'neutral', 'negative_affect_score': 0.0})
        calc.calculate_cli()
        
        # Reset
        calc.reset()
        
        assert calc.frames_processed == 0
        assert len(calc.gaze_history) == 0


class TestMetricScoring:
    """Tests for individual metric scoring"""
    
    def test_gaze_metrics_load_score(self):
        """Test gaze metrics produce valid load score"""
        metrics = GazeMetrics(
            fixation_duration=2.0,
            saccade_frequency=2.0,
            gaze_dispersion=0.5
        )
        score = metrics.get_load_score()
        assert 0 <= score <= 1
    
    def test_emotion_metrics_load_score(self):
        """Test emotion metrics produce valid load score"""
        metrics = EmotionMetrics(
            negative_affect_score=0.5,
            valence=0.3,
            arousal=0.7,
            brow_furrow_intensity=0.4
        )
        score = metrics.get_load_score()
        assert 0 <= score <= 1
    
    def test_mouse_metrics_load_score(self):
        """Test mouse metrics produce valid load score"""
        metrics = MouseMetrics(
            click_rate=1.5,
            path_efficiency=0.6,
            hover_dwell_time=1.0
        )
        score = metrics.get_load_score()
        assert 0 <= score <= 1


# Legacy test functions for backwards compatibility
def test_cognitive_load_monitor_initialization():
    """Test cognitive load monitor initialization (legacy)"""
    monitor = CognitiveLoadMonitor()
    assert monitor.gaze_wander_threshold == 0.3
    assert monitor.fixation_duration_threshold == 3.0
    assert len(monitor.gaze_history) == 0


def test_gaze_update():
    """Test gaze update (legacy)"""
    monitor = CognitiveLoadMonitor()
    monitor.update_gaze(0.5, 0.5)
    assert len(monitor.gaze_history) == 1
    assert monitor.gaze_history[0].x == 0.5
    assert monitor.gaze_history[0].y == 0.5


def test_gaze_wander_detection():
    """Test gaze wander detection (legacy)"""
    monitor = CognitiveLoadMonitor(gaze_wander_threshold=0.1)
    
    # Create wandering gaze pattern
    for i in range(20):
        x = (i % 5) * 0.2
        y = (i // 5) * 0.2
        monitor.update_gaze(x, y)
    
    assert monitor.detect_gaze_wander() == True


def test_fixation_duration_detection():
    """Test fixation duration detection (legacy)"""
    monitor = CognitiveLoadMonitor(fixation_duration_threshold=2.0)
    
    # Simulate long fixation
    start_time = datetime.now() - timedelta(seconds=3)
    monitor.current_fixation_start = start_time
    monitor.current_fixation_location = (0.5, 0.5)
    
    is_long, duration = monitor.detect_fixation_duration()
    assert is_long is True
    assert duration > 2.0


def test_cognitive_overload_check():
    """Test cognitive overload detection (legacy)"""
    monitor = CognitiveLoadMonitor()
    
    # Update with high negative affect
    emotion_result = {
        'negative_affect_score': 0.7,
        'emotions': {},
        'dominant_emotion': 'angry'
    }
    monitor.update_emotion(emotion_result)
    
    # Create wandering gaze
    for i in range(20):
        monitor.update_gaze(i * 0.05, i * 0.05)
    
    result = monitor.check_cognitive_overload()
    assert 'overload_detected' in result
    assert 'cognitive_load_score' in result
    assert 'triggers' in result
