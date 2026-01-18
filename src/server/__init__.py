"""
WebSocket server module for Adaptive UI System.

This module exposes the AdaptiveUISystem via WebSocket for browser extension integration.
"""

from .websocket_server import AdaptiveUIServer

__all__ = ["AdaptiveUIServer"]
