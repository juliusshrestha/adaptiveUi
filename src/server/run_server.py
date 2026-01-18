#!/usr/bin/env python3
"""
CLI entry point for the Adaptive UI WebSocket Server.

Usage:
    python -m src.server.run_server [--host HOST] [--port PORT] [--config CONFIG]

Examples:
    python -m src.server.run_server
    python -m src.server.run_server --port 9000
    python -m src.server.run_server --config custom_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.server.websocket_server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive UI WebSocket Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Start server on default port (8765):
        python -m src.server.run_server

    Start server on custom port:
        python -m src.server.run_server --port 9000

    Use custom config file:
        python -m src.server.run_server --config my_config.yaml

The server exposes the AdaptiveUISystem via WebSocket for browser extension
integration. Connect your browser extension to ws://localhost:8765 to receive
real-time cognitive load and adaptation data.
        """
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1 for localhost only)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           Adaptive UI WebSocket Server                         ║
╠═══════════════════════════════════════════════════════════════╣
║  Host:   {args.host:<52} ║
║  Port:   {args.port:<52} ║
║  Config: {args.config:<52} ║
╠═══════════════════════════════════════════════════════════════╣
║  Connect your browser extension to:                            ║
║  ws://{args.host}:{args.port:<48} ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    try:
        run_server(host=args.host, port=args.port, config_path=args.config)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
