import pandas_datareader as pdr # access fred
import pandas as pd
import requests # data from api
import plotly.express as px # visualize
from datetime import datetime

def get_fred_series_data(api_key,
                         series):
  # url
  url = "https://api.stlouisfed.org/geofred/series/data?series_id={0}&api_key={1}&file_type=json".format(series, api_key)
  # response
  response = requests.request("GET", url)
  return response

def transform_series_response(response):
  latest_date = list(response.json()['meta']['data'].keys())[0]
  return pd.DataFrame(response.json()['meta']['data'][latest_date])

def get_fred_data(param_list, start_date, end_date):
  df = pdr.DataReader(param_list, 'fred', start_date, end_date)
  return df.reset_index()

series = 'CABPPRIVSA' # https://fred.stlouisfed.org/series/CABPPRIVSA
  # get data for series
df = get_fred_data(param_list=['CABPPRIVSA'], 
                   start_date='2021-01-01', 
                   end_date='2022-12-31')
#df

fig = px.line(df, x="DATE", y="CABPPRIVSA", title='New Private Housing Units Authorized by Building Permits')
fig.show()

# get all series ids per series
response = get_fred_series_data(fred_api_key, series)
# transform response into a dataframe
df_all_series_ids = transform_series_response(response)
#df_all_series_ids.head()

# get all series to a list
series_list = df_all_series_ids['series_id'].tolist()
#print('Length of series list:', len(series_list) + 1)
#series_list[:5] # show first five in list

# set range for time
start_date = '2021-01-01'
end_date = datetime.today().strftime('%Y-%m-%d') # today

# get series data
df_permits_all_series = get_fred_data(param_list=series_list, # all series to get data for
                                      start_date=start_date, # start date
                                      end_date=end_date) # get latest date
#df_permits_all_series.head()

# transform columns to single column
df_melt = pd.melt(df_permits_all_series, id_vars=['DATE'], value_vars=series_list, var_name='STATE', value_name='PERMITS')
#df_melt.head()

# modify state abbreviation
df_plot = df_melt.copy() # copy df
df_plot['STATE'] = df_plot.apply(lambda x: x['STATE'][:2], axis=1)
#df_plot.head()

# plot
fig = px.line(df_plot, 
              x="DATE", # horizontal axis
              y="PERMITS", # vertical axis
              color='STATE', # split column
              title='New Private Housing Units Authorized by Building Permits')
fig.show()

# download file
#file_name = f'{series}_{start_date}-{end_date}.csv'
#df_plot.to_csv(file_name, index=False)
#files.download(file_name)
#print('Download {0}'.format(file_name))
