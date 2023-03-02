import pandas as pd
import graphing as graph
import statsmodels.formula.api as smf

dataset = pd.read_csv('https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-illness.csv', delimiter='\t')

graph.histogram(dataset, label_x='age', nbins=10, title='Feature', show=True)
graph.histogram(dataset, label_x='core_temperature', nbins=10, title='Label', show=True)
graph.scatter_2D(dataset, label_x='age', label_y='core_temperature', title='core tmeperature as a function of age', show=True)

formula = 'core_temperature ~ age'
model = smf.ols(formula = formula, data=dataset).fit()

graph.scatter_2D(dataset, 
                 label_x='age',
                 label_y='core_temperature', 
                 trendline=lambda x: model.params[1]*x+model.params[0],
                 show=True
                 )

print('Intercept:', model.params[0], 'Slope:', model.params[1])

graph.box_and_whisker(dataset, 'male', 'core_temperature', show=True)
graph.box_and_whisker(dataset, 'attended_training', 'core_temperature', show=True)
graph.box_and_whisker(dataset, 'ate_at_tonys_steakhouse', 'core_temperature', show=True)
graph.scatter_2D(dataset, 'body_fat_percentage', 'core_temperature', show=True)
graph.scatter_2D(dataset, 'protein_content_of_last_meal', 'core_temperature', show=True)
graph.scatter_2D(dataset, 'age', 'core_temperature', show=True)

for feature in ['male', 'age', 'protein_content_of_last_meal', 'body_fat_percentage']:
    formula = 'core_temperature ~' + feature
    simple_model = smf.ols(formula = formula, data = dataset).fit()
    print(feature)
    print('R-squared:', simple_model.rsquared)
    graph.scatter_2D(dataset, label_x=feature,
                     label_y='core_temperature',
                     title = feature,
                     trendline=lambda x: simple_model.params[1]*x + simple_model.params[0],
                     show=True)

# R-squared

formula = 'core_temperature ~ age'
age_trained_model = smf.ols(formula=formula, data=dataset).fit()
age_naive_model = smf.ols(formula=formula, data=dataset).fit()
age_naive_model.params[0] = dataset['core_temperature'].mean()
age_naive_model.params[1] = 0 

print('naive R-squared:', age_naive_model.rsquared)
print('trained R-squared:', age_trained_model.rsquared)

graph.scatter_2D(dataset, label_x='age',
                 label_y='core_temperature',
                 title='Naive model',
                 trendline=lambda x: dataset['core_temperature'].repeat(len(x)),
                 show=True)
graph.scatter_2D(dataset, label_x='age',
                 label_y='core_temperature',
                 title='Trained model',
                 trendline=lambda x: age_trained_model.params[1]*x + age_trained_model.params[0], 
                 show=True)

model = smf.ols(formula='core_temperature ~ age + male', data = dataset).fit()
print('R-squared:', model.rsquared)

import numpy as np
# Show a graph of the result
# this needs to be 3D, because we now have three variables in play: two features and one label

def predict(age, male):
    '''
    This converts given age and male values into a prediction from the model
    '''
    # to make a prediction with statsmodels, we need to provide a dataframe
    # so create a dataframe with just the age and male variables
    df = pd.DataFrame(dict(age=[age], male=[male]))
    return model.predict(df)

# Create the surface graph
fig = graph.surface(
    x_values=np.array([min(dataset.age), max(dataset.age)]),
    y_values=np.array([0, 1]),
    calc_z=predict,
    axis_title_x="Age",
    axis_title_y="Male",
    axis_title_z="Core temperature"
)

# Add our datapoints to it and display
fig.add_scatter3d(x=dataset.age, y=dataset.male, z=dataset.core_temperature, mode='markers')
fig.show()

model.summary()

age_trained_model.summary()
