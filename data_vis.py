# %%
from plotnine import ggplot, aes, geom_line, geom_point
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %%
from plotnine.data import mpg
ggplot(mpg, aes(x='class', y='hwy')) + geom_point()
