# PyGaze Setup Instructions

## Current Status

PyGaze is installed, but it requires PsychoPy, which has a dependency (`tables`) that's difficult to compile on macOS.

## Recommended Solution: Use Conda

The easiest way to install PyGaze with all dependencies is using conda:

```bash
# If you don't have conda, install Miniconda first:
# https://docs.conda.io/en/latest/miniconda.html

# Create a conda environment for this project
conda create -n adaptiveui python=3.9
conda activate adaptiveui

# Install PyGaze and PsychoPy (conda handles all dependencies)
conda install -c conda-forge psychopy python-pygaze

# Install other project dependencies
pip install -r requirements.txt
```

## Alternative: Skip PsychoPy (Limited Functionality)

If you just want to explore the code structure without full PyGaze functionality:

1. The code will detect missing PsychoPy and show helpful error messages
2. You can review all the code and examples
3. Install PsychoPy when you're ready to use it

## Current Error

When running examples, you'll see:
```
ImportError: PyGaze requires PsychoPy to be installed.
```

This is expected until PsychoPy is installed.

## Testing Without Full Installation

You can still:
- Review the code structure
- Read the documentation
- Understand the API design
- See how it integrates with cognitive load calculator

The wrapper classes are designed and ready - they just need PsychoPy to function.

## Next Steps

1. **Option A (Recommended)**: Use conda to install PsychoPy
2. **Option B**: Continue development without PyGaze for now
3. **Option C**: Use a different eye tracking library if needed

## Why Conda?

Conda pre-compiles packages like `tables` with all dependencies, avoiding compilation issues on macOS. It's the most reliable way to install PsychoPy.

## Verification

Once PsychoPy is installed via conda:

```bash
python -c "from pygaze import libscreen; print('PyGaze ready!')"
python examples/pygaze_basic_example.py
```

## Questions?

See `docs/PYGAZE_INSTALLATION.md` for more detailed troubleshooting.
