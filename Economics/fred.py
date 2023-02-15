import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 500)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]

from fredapi import Fred
from GetAPI import get_keys

secrets = get_keys('~/.secret/api-key.json')
fred_key = secrets['fred-api']
fred = Fred(api_key=fred_key)

sp_search = fred.search('S&P', order_by='popularity')
sp500 = fred.get_series(series_id='SP500')
sp500.plot(figsize=(10, 5), title='S&P 500', lw=2)
plt.show()