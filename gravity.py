import pandas as pd
import matplotlib.pyplot as plt

mydata = pd.read_excel("/Users/sangyong/Library/CloudStorage/Dropbox/Data Analysis/Gravity Model/gravity91.xlsx")

gg = plt.scatter(mydata['Country'], mydata['Export1'], c=mydata['Country'])
plt.xlabel('Country')
plt.ylabel('Export')
plt.title('Export')
plt.show()

gg1 = plt.scatter(mydata['quarter'], mydata['Export1'], c=mydata['quarter'])
plt.ylim(0, 2500000)
plt.xlabel('Quarter')
plt.ylabel('Export')
plt.title('Export')
plt.show()

