"""
Cognitive Load Metrics Collection
NASA-TLX, Task Completion Time, Error Frequency, Pupillary Dilation
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json


@dataclass
class TaskMetrics:
    """Container for task performance metrics"""
    task_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    completion_time: Optional[float] = None  # seconds
    errors: List[Dict] = field(default_factory=list)
    error_count: int = 0
    interactions: List[Dict] = field(default_factory=list)
    adaptation_triggers: List[Dict] = field(default_factory=list)
    
    def complete_task(self):
        """Mark task as complete and calculate completion time"""
        self.end_time = datetime.now()
        if self.start_time:
            self.completion_time = (self.end_time - self.start_time).total_seconds()
    
    def add_error(self, error_type: str, error_details: Dict):
        """Record an error"""
        error_record = {
            'type': error_type,
            'timestamp': datetime.now().isoformat(),
            'details': error_details
        }
        self.errors.append(error_record)
        self.error_count = len(self.errors)
    
    def add_interaction(self, interaction_type: str, interaction_details: Dict):
        """Record a user interaction"""
        interaction_record = {
            'type': interaction_type,
            'timestamp': datetime.now().isoformat(),
            'details': interaction_details
        }
        self.interactions.append(interaction_record)
    
    def add_adaptation_trigger(self, trigger_details: Dict):
        """Record an adaptation trigger"""
        trigger_record = {
            'timestamp': datetime.now().isoformat(),
            'details': trigger_details
        }
        self.adaptation_triggers.append(trigger_record)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'completion_time': self.completion_time,
            'error_count': self.error_count,
            'errors': self.errors,
            'interaction_count': len(self.interactions),
            'interactions': self.interactions,
            'adaptation_trigger_count': len(self.adaptation_triggers),
            'adaptation_triggers': self.adaptation_triggers
        }


class NASATLX:
    """
    NASA Task Load Index (TLX) questionnaire
    Measures perceived workload across 6 dimensions
    """
    
    DIMENSIONS = [
        'mental_demand',
        'physical_demand',
        'temporal_demand',
        'performance',
        'effort',
        'frustration'
    ]
    
    def __init__(self):
        """Initialize NASA-TLX"""
        self.ratings: Dict[str, Optional[int]] = {dim: None for dim in self.DIMENSIONS}
        self.pairwise_comparisons: Dict[str, int] = {}
        self.weighted_score: Optional[float] = None
    
    def set_rating(self, dimension: str, rating: int):
        """
        Set rating for a dimension (0-100 scale)
        
        Args:
            dimension: One of the 6 TLX dimensions
            rating: Rating value (0-100)
        """
        if dimension not in self.DIMENSIONS:
            raise ValueError(f"Invalid dimension: {dimension}")
        if not 0 <= rating <= 100:
            raise ValueError(f"Rating must be between 0 and 100, got {rating}")
        
        self.ratings[dimension] = rating
    
    def set_pairwise_comparison(self, dimension1: str, dimension2: str, winner: str):
        """
        Set pairwise comparison (which dimension contributes more to workload)
        
        Args:
            dimension1: First dimension
            dimension2: Second dimension
            winner: Which dimension wins ('dimension1' or 'dimension2')
        """
        if winner == 'dimension1':
            self.pairwise_comparisons[f"{dimension1}_{dimension2}"] = 1
        elif winner == 'dimension2':
            self.pairwise_comparisons[f"{dimension1}_{dimension2}"] = 0
        else:
            raise ValueError("Winner must be 'dimension1' or 'dimension2'")
    
    def calculate_weighted_score(self) -> float:
        """
        Calculate weighted TLX score
        
        Returns:
            Weighted TLX score (0-100)
        """
        # Check if all ratings are set
        if any(rating is None for rating in self.ratings.values()):
            raise ValueError("All ratings must be set before calculating weighted score")
        
        # Calculate weights from pairwise comparisons (simplified)
        # In full TLX, weights are calculated from all pairwise comparisons
        weights = {dim: 1.0 / len(self.DIMENSIONS) for dim in self.DIMENSIONS}
        
        # Calculate weighted score
        weighted_sum = sum(
            weights[dim] * self.ratings[dim]
            for dim in self.DIMENSIONS
        )
        
        self.weighted_score = weighted_sum
        return weighted_sum
    
    def get_raw_score(self) -> float:
        """
        Calculate raw (unweighted) TLX score
        
        Returns:
            Raw TLX score (0-100)
        """
        if any(rating is None for rating in self.ratings.values()):
            return 0.0
        
        return sum(self.ratings.values()) / len(self.DIMENSIONS)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'ratings': self.ratings,
            'raw_score': self.get_raw_score(),
            'weighted_score': self.weighted_score,
            'pairwise_comparisons': self.pairwise_comparisons
        }


class MetricsCollector:
    """
    Main metrics collection class
    Tracks all cognitive load metrics during task performance
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.current_task: Optional[TaskMetrics] = None
        self.completed_tasks: List[TaskMetrics] = []
        self.nasatlx_scores: List[NASATLX] = []
        self.pupil_dilation_data: List[Dict] = []
    
    def start_task(self, task_id: str) -> TaskMetrics:
        """
        Start tracking a new task
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            TaskMetrics object for the task
        """
        if self.current_task is not None:
            # Complete previous task if still active
            self.current_task.complete_task()
            self.completed_tasks.append(self.current_task)
        
        self.current_task = TaskMetrics(task_id=task_id, start_time=datetime.now())
        return self.current_task
    
    def end_task(self) -> Optional[TaskMetrics]:
        """
        End current task
        
        Returns:
            Completed TaskMetrics object
        """
        if self.current_task is None:
            return None
        
        self.current_task.complete_task()
        self.completed_tasks.append(self.current_task)
        completed = self.current_task
        self.current_task = None
        
        return completed
    
    def record_error(self, error_type: str, error_details: Dict = None):
        """Record an error in the current task"""
        if self.current_task is None:
            return
        
        if error_details is None:
            error_details = {}
        
        self.current_task.add_error(error_type, error_details)
    
    def record_interaction(self, interaction_type: str, interaction_details: Dict = None):
        """Record a user interaction"""
        if self.current_task is None:
            return
        
        if interaction_details is None:
            interaction_details = {}
        
        self.current_task.add_interaction(interaction_type, interaction_details)
    
    def record_adaptation_trigger(self, trigger_details: Dict):
        """Record an adaptation trigger"""
        if self.current_task is None:
            return
        
        self.current_task.add_adaptation_trigger(trigger_details)
    
    def add_nasatlx(self, nasatlx: NASATLX):
        """Add NASA-TLX score"""
        self.nasatlx_scores.append(nasatlx)
    
    def record_pupil_dilation(self, pupil_diameter: float, timestamp: Optional[datetime] = None):
        """
        Record pupillary dilation measurement
        
        Args:
            pupil_diameter: Pupil diameter in millimeters
            timestamp: Timestamp of measurement
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.pupil_dilation_data.append({
            'diameter': pupil_diameter,
            'timestamp': timestamp.isoformat()
        })
    
    def get_summary(self) -> Dict:
        """Get summary of all collected metrics"""
        task_summaries = [task.to_dict() for task in self.completed_tasks]
        
        if self.current_task:
            task_summaries.append(self.current_task.to_dict())
        
        avg_completion_time = None
        avg_error_count = None
        total_tasks = len(self.completed_tasks)
        
        if total_tasks > 0:
            completion_times = [t.completion_time for t in self.completed_tasks if t.completion_time]
            error_counts = [t.error_count for t in self.completed_tasks]
            
            if completion_times:
                avg_completion_time = sum(completion_times) / len(completion_times)
            if error_counts:
                avg_error_count = sum(error_counts) / len(error_counts)
        
        nasatlx_summary = None
        if self.nasatlx_scores:
            nasatlx_summary = {
                'count': len(self.nasatlx_scores),
                'avg_raw_score': sum(t.get_raw_score() for t in self.nasatlx_scores) / len(self.nasatlx_scores),
                'avg_weighted_score': sum(t.weighted_score for t in self.nasatlx_scores if t.weighted_score) / len([t for t in self.nasatlx_scores if t.weighted_score])
            }
        
        return {
            'total_tasks': total_tasks,
            'current_task_active': self.current_task is not None,
            'average_completion_time': avg_completion_time,
            'average_error_count': avg_error_count,
            'nasatlx_summary': nasatlx_summary,
            'pupil_dilation_samples': len(self.pupil_dilation_data),
            'task_summaries': task_summaries
        }
    
    def export_to_json(self, filepath: str):
        """Export all metrics to JSON file"""
        data = {
            'completed_tasks': [task.to_dict() for task in self.completed_tasks],
            'nasatlx_scores': [tlx.to_dict() for tlx in self.nasatlx_scores],
            'pupil_dilation_data': self.pupil_dilation_data,
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

