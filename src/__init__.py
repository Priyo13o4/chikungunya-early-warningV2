# Chikungunya Early Warning System
"""
Chikungunya Early Warning & Decision System (India)
A Bayesian hierarchical state-space model for district-level outbreak prediction.

Project Structure:
    src/
    ├── common/      - Shared utilities and constants
    ├── data/        - BLOCK 1: Data loading and cleaning
    ├── features/    - BLOCK 2: Feature engineering
    ├── labels/      - Label generation (outbreak definition)
    ├── models/      - BLOCK 3: All models (baselines + Bayesian)
    ├── evaluation/  - BLOCK 4: Validation & metrics
    ├── decision/    - BLOCK 5: Decision layer & actions
    └── visualization/ - Plotting utilities
"""

__version__ = "0.1.0"
__author__ = "Chikungunya EWS Team"
