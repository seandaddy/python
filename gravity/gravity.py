import pandas as pd
import matplotlib.pyplot as plt

mydata = pd.read_excel("gravity91.xlsx")

gg = plt.scatter(mydata['Country'], mydata['Export1'], c=mydata['Country'].astype('category').cat.codes, cmap='viridis')
plt.xlabel('Country')
plt.xlabel('Country')
plt.ylabel('Export')
plt.title('Export')
plt.show()

gg1 = plt.scatter(mydata['quarter'], mydata['Export1'], c=mydata['quarter'].astype('category').cat.codes, cmap='viridis') 
plt.ylim(0, 2500000)
plt.xlabel('Quarter')
plt.ylabel('Export')
plt.title('Export')
plt.show()
