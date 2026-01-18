"""
WebSocket server for Adaptive UI System.

Exposes the AdaptiveUISystem via WebSocket for browser extension integration.
Runs frame processing in a background thread and broadcasts results to connected clients.
"""

import asyncio
import json
import threading
import time
from typing import Optional, Set
from dataclasses import dataclass, asdict
from queue import Queue, Empty

import cv2
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

import sys
from pathlib import Path


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Ensure imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.main import AdaptiveUISystem
from src.utils.logger import setup_logger


@dataclass
class AdaptationMessage:
    """Message structure for adaptation updates sent to clients."""
    type: str = "adaptation_update"
    timestamp: float = 0.0
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def to_json(self) -> str:
        return json.dumps(asdict(self), cls=NumpyJSONEncoder)


@dataclass
class StatusMessage:
    """Message structure for server status updates."""
    type: str = "status"
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}

    def to_json(self) -> str:
        return json.dumps(asdict(self), cls=NumpyJSONEncoder)


class AdaptiveUIServer:
    """
    WebSocket server that wraps AdaptiveUISystem.

    Runs frame processing in a background thread and broadcasts
    adaptation results to all connected browser extension clients.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        config_path: str = "config/config.yaml"
    ):
        """
        Initialize the WebSocket server.

        Args:
            host: Host address to bind to (default localhost only for security)
            port: Port to listen on
            config_path: Path to AdaptiveUISystem config
        """
        self.host = host
        self.port = port
        self.config_path = config_path

        # Setup logging
        self.logger = setup_logger(
            name="adaptive_ui_server",
            log_level="INFO",
            log_dir="logs",
            console_output=True
        )

        # WebSocket state
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server: Optional[websockets.WebSocketServer] = None

        # Processing state
        self.adaptive_system: Optional[AdaptiveUISystem] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.result_queue: Queue = Queue(maxsize=1)  # Only keep latest result
        self.running = False
        self.paused = False

        # Stats
        self.frames_processed = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0

        # Latest frame cache (for center calibration commands)
        self._last_frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None

    def _init_adaptive_system(self):
        """Initialize the AdaptiveUISystem."""
        self.logger.info("Initializing AdaptiveUISystem...")
        self.adaptive_system = AdaptiveUISystem(config_path=self.config_path)
        self.adaptive_system.initialize()
        self.logger.info("AdaptiveUISystem initialized successfully")

    def _process_frames(self):
        """
        Background thread that continuously processes camera frames.
        Results are placed in a queue for the async broadcast loop.
        """
        self.logger.info("Starting frame processing thread...")

        camera = self.adaptive_system.camera
        target_fps = self.adaptive_system.target_fps
        target_interval = 1.0 / target_fps
        last_frame_time = 0

        while self.running:
            # Pause handling
            if self.paused:
                time.sleep(0.1)
                continue

            # Frame rate control
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
                continue

            last_frame_time = current_time

            # Read and process frame
            ret, frame = camera.read()
            if not ret:
                self.logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue

            # Mirror the frame (consistent with main.py)
            frame = cv2.flip(frame, 1)

            # Store latest frame for commands like calibrate_center
            with self._last_frame_lock:
                self._last_frame = frame

            # Process through AdaptiveUISystem
            try:
                result = self.adaptive_system.process_frame(frame)
                self.frames_processed += 1
                self.fps_frame_count += 1

                # Calculate FPS every second
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = self.fps_frame_count / (current_time - self.last_fps_time)
                    self.fps_frame_count = 0
                    self.last_fps_time = current_time

                # Create message for clients
                message = self._create_adaptation_message(result)

                # Put in queue (replace old if full)
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    pass
                self.result_queue.put(message)

            except Exception as e:
                self.logger.error(f"Error processing frame: {e}", exc_info=True)

        self.logger.info("Frame processing thread stopped")

    def _create_adaptation_message(self, result: dict) -> AdaptationMessage:
        """
        Convert processing result to a WebSocket message.

        Args:
            result: Output from AdaptiveUISystem.process_frame()

        Returns:
            AdaptationMessage ready to send to clients
        """
        # Extract gaze coordinates
        gaze_coords = result.get('gaze_coords')
        if gaze_coords is not None:
            gaze_coords = [float(gaze_coords[0]), float(gaze_coords[1])]

        # Extract emotion data
        emotion = result.get('emotion', {})
        emotion_data = {
            'dominant': emotion.get('dominant_emotion', 'unknown'),
            'negative_affect_score': float(emotion.get('negative_affect_score', 0.0)),
            'is_stressed': emotion.get('is_stressed', False),
            'face_detected': emotion.get('face_detected', False)
        }

        # Extract cognitive load data
        overload = result.get('overload_status', {})
        cognitive_load_data = {
            'score': float(overload.get('cli', overload.get('cognitive_load_score', 0.0))),
            'level': overload.get('load_level', 'unknown'),
            'overload_detected': overload.get('overload_detected', False),
            'gaze_score': float(overload.get('gaze_score', 0.0)),
            'emotion_score': float(overload.get('emotion_score', 0.0)),
            'mouse_score': float(overload.get('mouse_score', 0.0)),
            'triggers': overload.get('triggers', {})
        }

        # Extract adaptation commands
        adaptation_commands = result.get('adaptation_commands', {})
        if not adaptation_commands:
            adaptation_commands = {
                'simplify': False,
                'guidance': False,
                'layout': False,
                'parameters': {}
            }

        return AdaptationMessage(
            type="adaptation_update",
            timestamp=time.time(),
            data={
                'frame_count': result.get('frame_count', 0),
                'gaze_coords': gaze_coords,
                'cognitive_load': cognitive_load_data,
                'emotion': emotion_data,
                'adaptation_commands': adaptation_commands,
                'adaptations': result.get('adaptations', [])
            }
        )

    def _create_status_message(self) -> StatusMessage:
        """Create a status message with current server state."""
        return StatusMessage(
            type="status",
            data={
                'connected': True,
                'camera_active': self.adaptive_system is not None and self.adaptive_system.camera is not None,
                'fps': round(self.fps, 1),
                'frames_processed': self.frames_processed,
                'clients_connected': len(self.clients),
                'paused': self.paused,
                'calibrated': self.adaptive_system.use_calibration if self.adaptive_system else False
            }
        )

    async def _handle_client(self, websocket: WebSocketServerProtocol):
        """
        Handle a connected client.

        Args:
            websocket: The client's WebSocket connection
        """
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.logger.info(f"Client connected: {client_id}")
        self.clients.add(websocket)

        try:
            # Send initial status
            status = self._create_status_message()
            await websocket.send(status.to_json())

            # Handle incoming messages from client
            async for message in websocket:
                await self._handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.discard(websocket)
            self.logger.info(f"Client removed: {client_id} (Total clients: {len(self.clients)})")

    async def _handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """
        Handle incoming message from a client.

        Args:
            websocket: The client's WebSocket connection
            message: The message received
        """
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == 'command':
                command = data.get('command')
                await self._handle_command(websocket, command)
            elif msg_type == 'config':
                config = data.get('config', {})
                await self._handle_config(websocket, config)
            elif msg_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong', 'timestamp': time.time()}))
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON received: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")

    async def _handle_command(self, websocket: WebSocketServerProtocol, command: str):
        """Handle a command from a client."""
        self.logger.info(f"Received command: {command}")

        if command == 'calibrate':
            # Note: Calibration requires display, so we log a warning
            self.logger.warning("Calibration requested but not supported in headless mode")
            await websocket.send(json.dumps({
                'type': 'command_response',
                'command': command,
                'success': False,
                'message': 'Calibration not supported in server mode'
            }))
        elif command == 'calibrate_center':
            # Center calibration for monitor-plane tracker
            success = False
            message = 'Not ready'
            if self.adaptive_system and hasattr(self.adaptive_system.gaze_tracker, "calibrate_center"):
                with self._last_frame_lock:
                    frame = self._last_frame.copy() if self._last_frame is not None else None
                if frame is not None:
                    try:
                        success = bool(self.adaptive_system.gaze_tracker.calibrate_center(frame))  # type: ignore[attr-defined]
                        message = 'Center calibrated' if success else 'Center calibration failed'
                    except Exception as e:
                        success = False
                        message = f'Center calibration error: {e}'
                else:
                    message = 'No camera frame available yet'
            else:
                message = 'Center calibration not supported by current gaze tracker'

            await websocket.send(json.dumps({
                'type': 'command_response',
                'command': command,
                'success': success,
                'message': message
            }))
        elif command == 'pause':
            self.paused = True
            await websocket.send(json.dumps({
                'type': 'command_response',
                'command': command,
                'success': True
            }))
        elif command == 'resume':
            self.paused = False
            await websocket.send(json.dumps({
                'type': 'command_response',
                'command': command,
                'success': True
            }))
        elif command == 'status':
            status = self._create_status_message()
            await websocket.send(status.to_json())
        else:
            self.logger.warning(f"Unknown command: {command}")

    async def _handle_config(self, websocket: WebSocketServerProtocol, config: dict):
        """Handle configuration update from a client."""
        self.logger.info(f"Received config update: {config}")

        # Apply config changes to the cognitive load monitor
        if self.adaptive_system and self.adaptive_system.cognitive_load_monitor:
            monitor = self.adaptive_system.cognitive_load_monitor

            if 'sensitivity' in config:
                # Map sensitivity (0-1) to threshold (0.8-0.5)
                sensitivity = config['sensitivity']
                threshold = 0.8 - (sensitivity * 0.3)
                monitor.cli_overload_threshold = threshold
                self.logger.info(f"Updated CLI threshold to {threshold}")

        # Allow switching gaze tracker mode at runtime (extension feature)
        gaze_mode = config.get('gaze_mode') or config.get('gazeMode')
        if gaze_mode and self.adaptive_system:
            # Pause frame processing during swap
            was_paused = self.paused
            self.paused = True
            time.sleep(0.05)
            mp_cfg = config.get('monitor_plane') or {}
            ok = self.adaptive_system.set_gaze_mode(str(gaze_mode), monitor_plane_config=mp_cfg)
            self.paused = was_paused
            if ok:
                self.logger.info(f"Switched gaze mode to {gaze_mode}")
            else:
                self.logger.warning(f"Failed to switch gaze mode to {gaze_mode}")

        await websocket.send(json.dumps({
            'type': 'config_response',
            'success': True,
            'applied': config
        }))

    async def _broadcast_loop(self):
        """
        Continuously broadcast frame processing results to all connected clients.
        """
        self.logger.info("Starting broadcast loop...")

        while self.running:
            try:
                # Get latest result from queue (non-blocking)
                try:
                    message = self.result_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.01)  # Small sleep to prevent busy loop
                    continue

                # Broadcast to all connected clients
                if self.clients:
                    message_json = message.to_json()

                    # Create tasks for all clients
                    tasks = [
                        asyncio.create_task(self._safe_send(client, message_json))
                        for client in self.clients.copy()
                    ]

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                self.logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(0.1)

        self.logger.info("Broadcast loop stopped")

    async def _safe_send(self, websocket: WebSocketServerProtocol, message: str):
        """
        Safely send a message to a client, handling disconnections.

        Args:
            websocket: The client's WebSocket connection
            message: The message to send
        """
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
        except Exception as e:
            self.logger.debug(f"Error sending to client: {e}")
            self.clients.discard(websocket)

    async def start(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting Adaptive UI WebSocket Server on ws://{self.host}:{self.port}")

        # Initialize the adaptive system
        self._init_adaptive_system()

        # Start frame processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()

        # Start WebSocket server
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )

        self.logger.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
        self.logger.info("Press Ctrl+C to stop")

        # Start broadcast loop
        broadcast_task = asyncio.create_task(self._broadcast_loop())

        try:
            # Keep server running
            await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            broadcast_task.cancel()
            await self.stop()

    async def stop(self):
        """Stop the WebSocket server and cleanup resources."""
        self.logger.info("Stopping server...")

        self.running = False

        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        # Close all client connections
        if self.clients:
            close_tasks = [client.close() for client in self.clients]
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Cleanup adaptive system
        if self.adaptive_system:
            self.adaptive_system.cleanup()

        self.logger.info("Server stopped")


def run_server(host: str = "127.0.0.1", port: int = 8765, config_path: str = "config/config.yaml"):
    """
    Run the WebSocket server.

    Args:
        host: Host address to bind to
        port: Port to listen on
        config_path: Path to config file
    """
    server = AdaptiveUIServer(host=host, port=port, config_path=config_path)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user")


if __name__ == "__main__":
    run_server()
