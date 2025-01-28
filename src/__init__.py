import os
import random

import numpy
import torch

random.seed(42)
torch.manual_seed(42)
numpy.random.seed(42)

MUSDB_PATH = os.environ.get("MUSDB_PATH")
MOISESDB_PATH = os.environ.get("MOISESDB_PATH")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
