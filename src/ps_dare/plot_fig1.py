import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .paths import solve_path

# load the data
PREDICTIONS_FILE = "data/weekly/predictions/" "general_giustiniani.csv"

df = pd.read_csv(solve_path(PREDICTIONS_FILE))