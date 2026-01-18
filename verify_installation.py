#!/usr/bin/env python3
"""
Verification script to check if all dependencies are installed correctly
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} - OK")
        return True
    except ImportError as e:
        print(f"✗ {package_name} - FAILED: {e}")
        return False

def main():
    """Check all required dependencies"""
    print("Checking Adaptive UI System Dependencies...")
    print("=" * 50)

    print(f"Python: {sys.version.split()[0]}")
    print()

    required = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("yaml", "PyYAML"),
        ("websockets", "websockets"),
        ("dateutil", "python-dateutil"),
    ]

    optional = [
        ("pynput", "pynput (mouse tracking)"),
        ("torch", "PyTorch (ViT emotion model)"),
        ("transformers", "Transformers (ViT emotion model)"),
        ("tensorflow", "TensorFlow (optional)"),
        ("fer", "FER (optional emotion model)"),
    ]

    results = []
    print("Required:")
    for module, name in required:
        results.append(check_import(module, name))

    print()
    print("Optional:")
    for module, name in optional:
        check_import(module, name)

    print("=" * 50)

    if all(results):
        print("\n✓ Required dependencies installed successfully!")
        print("\nRun desktop app:")
        print("  python main.py")
        print("\nRun WebSocket server (for extension):")
        print("  python -m src.server.run_server")
        return 0

    print("\n✗ Missing required dependencies.")
    print("Install minimal runtime with:")
    print("  pip install -r requirements/base.txt")
    print("\nOr install the full set with:")
    print("  pip install -r requirements.txt")
    return 1

if __name__ == "__main__":
    sys.exit(main())

