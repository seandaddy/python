import pandas as pd
import graphing as graph
import statsmodels.formula.api as smf
import numpy as np

dataset = pd.read_csv('https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-illness.csv', delimiter='\t')

simple_formula = 'core_temperature ~ protein_content_of_last_meal'
simple_model = smf.ols(formula = simple_formula, data=dataset).fit()

graph.scatter_2D(dataset, label_x='protein_content_of_last_meal',
                 label_y='core_temperature',
                 trendline=lambda x: simple_model.params[1]*x + simple_model.params[0], show=True)

print('R-squared:', simple_model.rsquared)

# Simple Polynomial Regression
polynomial_formula = 'core_temperature ~ protein_content_of_last_meal + I(protein_content_of_last_meal**2)'
polynomial_model = smf.ols(formula= polynomial_formula, data=dataset).fit()

graph.scatter_2D(dataset, label_x='protein_content_of_last_meal',
                 label_y='core_temperature',
                 trendline=lambda x: polynomial_model.params[2]*x**2 + polynomial_model.params[1]*x + polynomial_model.params[0], show=True)

print('R-squared:', polynomial_model.rsquared)

fig = graph.surface(
        x_values=np.array([min(dataset.protein_content_of_last_meal), max(dataset.protein_content_of_last_meal)]),
        y_values=np.array([min(dataset.protein_content_of_last_meal)**2, max(dataset.protein_content_of_last_meal)**2]),
        calc_z=lambda x,y: polynomial_model.params[0] + (polynomial_model.params[1] * x) + (polynomial_model.params[2] * y),
        axis_title_x="x",
        axis_title_y="x2",
        axis_title_z="Core temperature"
        )
fig.add_scatter3d(x=dataset.protein_content_of_last_meal, y=dataset.protein_content_of_last_meal**2, z=dataset.core_temperature, mode='markers')
fig.show()

graph.scatter_2D(dataset, label_x='protein_content_of_last_meal',
                 label_y='core_temperature',
                 x_range=[0,100],
                 trendline=lambda x: simple_model.params[1]*x + simple_model.params[0], show=True)

graph.scatter_2D(dataset, label_x='protein_content_of_last_meal',
                 label_y='core_temperature',
                 x_range=[0,100],
                 trendline=lambda x: polynomial_model.params[2]*x**2 + polynomial_model.params[1]*x + polynomial_model.params[0], show=True)

