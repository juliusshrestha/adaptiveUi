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
    
    checks = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("yaml", "PyYAML"),
        ("dateutil", "python-dateutil"),
        ("pytest", "pytest"),
        ("mediapipe", "MediaPipe"),
        ("tensorflow", "TensorFlow"),
    ]
    
    results = []
    for module, name in checks:
        results.append(check_import(module, name))
    
    print("=" * 50)
    
    if all(results):
        print("\n✓ All dependencies installed successfully!")
        print("\nYou can now run the system with:")
        print("  python src/main.py")
        return 0
    else:
        print("\n✗ Some dependencies are missing.")
        print("Please install them with:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

