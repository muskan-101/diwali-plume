import sys
import numpy as np
import pandas as pd
import cv2
import torch
import streamlit
import xgboost
import lightgbm

print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nDIWALI PLUME ENVIRONMENT READY")