"""
UI Adaptation Module
Triggers UI changes based on cognitive load:
- Simplification: Hiding non-essential elements
- Guidance: Highlighting next steps or providing tooltips
- Layout Reorganization: Increasing whitespace or font size
"""

from typing import Dict, List, Optional
from enum import Enum


class AdaptationType(Enum):
    """Types of UI adaptations"""
    SIMPLIFICATION = "simplification"
    GUIDANCE = "guidance"
    LAYOUT_REORGANIZATION = "layout_reorganization"
    FONT_SIZE_INCREASE = "font_size_increase"
    WHITESPACE_INCREASE = "whitespace_increase"
    HIDE_DECORATIVE = "hide_decorative"
    SHOW_TOOLTIPS = "show_tooltips"
    HIGHLIGHT_NEXT_STEP = "highlight_next_step"


class UIAdapter:
    """
    Manages UI adaptations based on cognitive load triggers
    """
    
    def __init__(self):
        """Initialize UI adapter"""
        self.active_adaptations: List[AdaptationType] = []
        self.adaptation_history: List[Dict] = []
    
    def generate_adaptations(self, overload_status: Dict[str, any]) -> List[AdaptationType]:
        """
        Generate list of adaptations based on cognitive overload status
        
        Args:
            overload_status: Result from cognitive_load_monitor.check_cognitive_overload()
            
        Returns:
            List of adaptation types to apply
        """
        adaptations = []
        
        if not overload_status.get('overload_detected', False):
            return adaptations
        
        triggers = overload_status.get('triggers', {})
        load_score = overload_status.get('cognitive_load_score', 0.0)
        
        # Always apply simplification for high cognitive load
        adaptations.append(AdaptationType.SIMPLIFICATION)
        adaptations.append(AdaptationType.HIDE_DECORATIVE)
        
        # If gaze wander detected, provide guidance
        if triggers.get('gaze_wander', False):
            adaptations.append(AdaptationType.GUIDANCE)
            adaptations.append(AdaptationType.HIGHLIGHT_NEXT_STEP)
            adaptations.append(AdaptationType.SHOW_TOOLTIPS)
        
        # If long fixation detected, simplify layout
        if triggers.get('fixation_duration', False):
            adaptations.append(AdaptationType.LAYOUT_REORGANIZATION)
            adaptations.append(AdaptationType.WHITESPACE_INCREASE)
            adaptations.append(AdaptationType.FONT_SIZE_INCREASE)
        
        # If negative affect detected, increase simplification
        if triggers.get('negative_affect', False):
            adaptations.append(AdaptationType.SIMPLIFICATION)
            adaptations.append(AdaptationType.GUIDANCE)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_adaptations = []
        for adapt in adaptations:
            if adapt not in seen:
                seen.add(adapt)
                unique_adaptations.append(adapt)
        
        self.active_adaptations = unique_adaptations
        return unique_adaptations
    
    def get_adaptation_parameters(self, adaptation_type: AdaptationType) -> Dict[str, any]:
        """
        Get parameters for specific adaptation type
        
        Args:
            adaptation_type: Type of adaptation
            
        Returns:
            Dictionary with adaptation parameters
        """
        params = {
            AdaptationType.SIMPLIFICATION: {
                'hide_non_essential': True,
                'reduce_visual_elements': 0.5
            },
            AdaptationType.GUIDANCE: {
                'show_guidance': True,
                'guidance_intensity': 'high'
            },
            AdaptationType.LAYOUT_REORGANIZATION: {
                'increase_spacing': True,
                'spacing_multiplier': 1.5
            },
            AdaptationType.FONT_SIZE_INCREASE: {
                'font_size_multiplier': 1.2,
                'min_font_size': 14
            },
            AdaptationType.WHITESPACE_INCREASE: {
                'whitespace_multiplier': 1.5,
                'min_margin': 20
            },
            AdaptationType.HIDE_DECORATIVE: {
                'hide_decorative_elements': True,
                'hide_graphics': True
            },
            AdaptationType.SHOW_TOOLTIPS: {
                'show_tooltips': True,
                'tooltip_duration': 5.0
            },
            AdaptationType.HIGHLIGHT_NEXT_STEP: {
                'highlight_enabled': True,
                'highlight_color': '#FFD700',
                'highlight_animation': True
            }
        }
        
        return params.get(adaptation_type, {})
    
    def apply_adaptations(self, adaptations: List[AdaptationType]) -> Dict[str, any]:
        """
        Generate adaptation commands for UI
        
        Args:
            adaptations: List of adaptation types to apply
            
        Returns:
            Dictionary with adaptation commands
        """
        adaptation_commands = {
            'simplify': False,
            'guidance': False,
            'layout': False,
            'parameters': {}
        }
        
        for adapt_type in adaptations:
            params = self.get_adaptation_parameters(adapt_type)
            
            if adapt_type == AdaptationType.SIMPLIFICATION:
                adaptation_commands['simplify'] = True
                adaptation_commands['parameters'].update(params)
            
            elif adapt_type == AdaptationType.GUIDANCE:
                adaptation_commands['guidance'] = True
                adaptation_commands['parameters'].update(params)
            
            elif adapt_type == AdaptationType.LAYOUT_REORGANIZATION:
                adaptation_commands['layout'] = True
                adaptation_commands['parameters'].update(params)
            
            else:
                # Merge other parameters
                adaptation_commands['parameters'].update(params)
        
        # Log adaptation
        self.adaptation_history.append({
            'adaptations': [a.value for a in adaptations],
            'commands': adaptation_commands,
            'timestamp': None  # Would add timestamp in production
        })
        
        return adaptation_commands
    
    def clear_adaptations(self):
        """Clear all active adaptations"""
        self.active_adaptations = []
        return {
            'simplify': False,
            'guidance': False,
            'layout': False,
            'parameters': {}
        }

