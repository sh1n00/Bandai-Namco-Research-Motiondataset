import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./dataset-1_bow_angry_001_pos.csv")
x, y = [], []
for i in range(1, len(range(df.shape[1])), 3):
    x.append(df.iloc[1, i])
    y.append(df.iloc[1, i+1])

plt.scatter(x, y)
plt.show()