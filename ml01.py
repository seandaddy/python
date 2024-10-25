import pandas as pd
import graphing
import statsmodels.formula.api as smf

data = pd.read_csv('https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv')
dataset = pd.DataFrame(data)
#print(dataset)

formula = "boot_size ~ harness_size"
model = smf.ols(formula = formula, data = dataset)

fitted_model = model.fit()
print("The following model parameters have been found:\n" +
        f"Line slope: {fitted_model.params[1]}\n"+
        f"Line Intercept: {fitted_model.params[0]}")
graphing.scatter_2D(dataset,
                    label_x="harness_size",
                    label_y="boot_size",
                    trendline=lambda x: fitted_model.params[1] * x + fitted_model.params[0],
                    show=True
                    )

harness_size = { 'harness_size' : [52.5] }
approximate_boot_size = fitted_model.predict(harness_size)

print("Estimated approximate_boot_size: ", approximate_boot_size[0])
