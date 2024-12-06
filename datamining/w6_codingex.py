# %%
import tensorflow as tf
with tf.device('/device:GPU:0'):
    pass

# %%
import os
import pandas as pd
import numpy as np
from pandas.core.internals.blocks import Categorical
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.iolib.summary2 import summary_col
from sklearn.linear_model import LogisticRegression

# %%
df = pd.read_stata('/Users/eer/Documents/python/data/DAAN545_GlassCliff.dta')
with pd.option_context('display.max_columns', 40):
    df.describe()

# %%
y = df['female']
dummies = pd.get_dummies(df[['year_begin','sich_2digits_group']].astype(int).astype(str), drop_first=True)

# %%
# List of related variables
vars_Return = ["roa_bl_2_1","roe_bl_2_1","salesgrowth_bl_2_1","tobin_q_1","leverage_1"] # Market return of the firm
vars_ProbDefault  = ["pdefault_bshum_q1","p_altman_1"] # Probability of default

vars_Prior   = ["diff_15_4", "idio_vol_15_4"] # Prior performance of the firm
vars_CAPM    = ["alpha_capm_15_4","beta_capm_15_4"] # CAPM
vars_FF3     = ["alpha_ff3_15_4", "beta_ff3_market_15_4","beta_ff3_size_15_4","beta_ff3_value_15_4"] # Fama-French 3 factors
vars_FF4     = ["alpha_ff4_15_4",  "beta_ff4_market_15_4","beta_ff4_size_15_4","beta_ff4_value_15_4","beta_ff4_mom_15_4"]	 # Fama-French 4 factors
vars_AQR5    = ["alpha_aqr5_15_4", "beta_aqr5_market_15_4","beta_aqr5_size_15_4","beta_aqr5_value_15_4","beta_aqr5_mom_15_4","beta_aqr5_quality_15_4"] # AQR 5 factors

# %%
# Firt, let's see how the performance affects the leader's gender.
main_results = []
for var in vars_Return:
    X_ = pd.concat([dummies, df[[var,'lagfemaleratio']] ], axis=1).astype(float)
    X_ = add_constant(X_)
    result = Logit(y, X_).fit()
    main_results.append(result)
X_ = pd.concat([dummies, df[vars_Return+['lagfemaleratio']] ], axis=1).astype(float)
X_ = add_constant(X_)
result = Logit(y, X_).fit()
main_results.append(result)
summary_col(main_results,stars=True,regressor_order=vars_Return+['lagfemaleratio'])

# %%
# Compare with sklearn
X_ = pd.concat([dummies, df[vars_Return+['lagfemaleratio']] ], axis=1)
X_ = add_constant(X_)
model = LogisticRegression(max_iter=1000)
result = model.fit(X_, y)
coef = pd.Series(result.coef_[0], index=X_.columns)
print(coef)

# Check the average marginal effect of sklearn model
from sklearn.inspection import partial_dependence
tes=partial_dependence(result, X_, features=['lagfemaleratio'])

# %%
# Average partial effect
ape_results = []
for result in main_results:
    ape_results.append(result.get_margeff(at='mean', method='dydx', atexog=None, dummy=False, count=False))
ape_results[0].summary()

# %%
additional_results = []
for vars in [vars_ProbDefault, vars_Prior, vars_CAPM, vars_FF3, vars_FF4, vars_AQR5]:
    X_ = pd.concat([dummies, df[vars+['lagfemaleratio']] ], axis=1).astype(float)
    X_ = add_constant(X_)
    result = Logit(y, X_).fit()
    additional_results.append(result)
summary_col(additional_results,stars=True,regressor_order=vars_Prior+vars_CAPM+vars_FF3+vars_FF4+vars_AQR5+['lagfemaleratio'])

# %%
# One issue is that "female" is very imbalanced.
print(y.mean())

# %%
# Let's try to balance the data
from imblearn.over_sampling import SMOTENC
for i in range(10):
    smote = SMOTENC(categorical_features=['gvkey_num','year_begin','sich_2digits_group'])
    X = df.drop(columns=['female']).astype(float)
    X_res, y_res = smote.fit_resample(X, y)
    df_res = pd.concat([X_res, y_res], axis=1)
    df_res.to_stata(f'GlassCliff_smote{i+1}.dta')
df_res

# %%
for i in range(10):
    smote = SMOTENC(categorical_features=['year_begin','sich_2digits_group'])
    X = df.drop(columns=['female'])
    X_res, y_res = smote.fit_resample(X, y)
    dummies_res = pd.get_dummies(X_res[['year_begin','sich_2digits_group']].astype(int).astype(str), drop_first=True)
    fit_args = {'maxiter':100,'cov_type':'cluster','cov_kwds':{'groups': X_res['gvkey_num']}}
    main_results_smote = []
    for var in vars_Return:
        print(var)
        X_ = pd.concat([dummies_res,X_res[[var,'lagfemaleratio']] ], axis=1) .astype(float)
        X_ = add_constant(X_)
        result = Logit(y_res, X_).fit(**fit_args)
        main_results_smote.append(result)
    X_ = pd.concat([dummies_res, X_res[vars_Return+['lagfemaleratio']] ], axis=1).astype(float)
    X_ = add_constant(X_)
    result = Logit(y_res, X_).fit(**fit_args)
    main_results_smote.append(result)
    print(summary_col(main_results_smote,stars=True,regressor_order=vars_Return+['lagfemaleratio']))

# %%
