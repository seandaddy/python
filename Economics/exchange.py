import pandas as pd
from itertools import cycle
import matplotlib.pylab as plt

plt.style.use("seaborn-whitegrid")
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

to_usd_df = pd.read_csv('~/Documents/python/Economics/exchange_rate_to_usd.csv')

for exrate in to_usd_df.columns:
    if exrate == 'date':
        continue
    to_usd_df[exrate] \
        .plot(figsize=(15, 5), title=exrate, color=next(color_cycle))
    plt.show()
