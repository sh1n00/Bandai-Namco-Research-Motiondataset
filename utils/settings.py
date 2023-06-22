import os
import torch

ROOT_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")

seed = 42

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
