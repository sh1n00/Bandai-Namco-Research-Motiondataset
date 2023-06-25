import os
import torch

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

RESULT_DIR = os.path.join(ROOT_DIR, "result")

BUFFER = 20

seed = 42

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
