import os

import pandas as pd

from utils import animation, settings

filename = "dataset-1_run_active_001_pos.csv"
df_label = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "action", "run", filename))
df_label.drop("action", axis=1, inplace=True)
df_label = df_label.iloc[:20, :]
animation.create_gif(df_label, "dataset-1_run_active_001_pos_20frame.csv", True)
df_label.head()
