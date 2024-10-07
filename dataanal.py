import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("StudentsPerformance.csv")
df.head()

plt.style.use('dark_background')
sns.set_theme(style="darkgrid",palette="bright", font_scale=1.5)
sns.pairplot(df[['math score','reading score','writing score']], height=4)

def average_score(dt): return (dt['math score']+dt['reading score']+dt['writing score'])/3
df['average score'] = df.apply(average_score, axis=1)
df.head()

sns.catplot(x='lunch', y='average score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('average score')

sns.catplot(x='lunch', y='math score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('math')
sns.catplot(x='lunch', y='reading score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('reading')
sns.catplot(x='lunch', y='writing score', hue='gender', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('writing')

sns.catplot(x='test preparation course', y='average score', kind='boxen', data=df, height=10, palette=sns.color_palette(['red', 'blue']))
plt.title('average')

sns.catplot(x='parental level of education', y='average score', kind='boxen', data=df, height=14)
plt.title('average') 
plt.legend(loc='lower right')

sns.catplot(x='race/ethnicity', y='average score', kind='boxen', data=df, height=14)
plt.title('average')
plt.legend(loc='lower right')
plt.show()
