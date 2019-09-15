# local imports
from cleaning import DataSet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train = DataSet("train_values.csv", "train_labels.csv")

train_full = pd.concat([train.values_df, train.labels_series], axis=1)

g=sns.FacetGrid(train_full, col="geo_level_1_id")
g.map(plt.hist, "damage_grade")

plt.show()
