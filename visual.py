import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('penguins')

plt.scatter(df['flipper_length_mm'], df['body_mass_g'])
plt.xlabel('Flipper Length')
plt.ylabel('Body Mass')
plt.show()

df_group = df.groupby('species')['body_mass_g'].mean().reset_index()
plt.bar(x=df_group['species'], height=df_group['body_mass_g'])
plt.xlabel('Species')
plt.ylabel('Body Mass')
plt.show()

plt.hist(df['body_mass_g'], bins=30)
plt.xlabel('Body Mass')
plt.ylabel('Count')
plt.title('Weight Distribution of Penguins')
plt.show()

