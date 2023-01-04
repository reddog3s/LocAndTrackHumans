import pandas as pd
import numpy as np


times = pd.read_csv('times.csv')
print(times.head())

times = times.to_numpy()
print(np.nanmean(times))


