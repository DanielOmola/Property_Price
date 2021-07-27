# ##################################
#
#  much of the cod is stored in 
#  my package for more readability
#
# ##################################
#from mypackage import ploter_bis as plt_bis
from mypackage import ploter
from mypackage import data_processor as dp
from mypackage import mydataloader as dl
import os



import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import dump
import platform



####################################
# 		1 - Data Preparation
####################################

# ////////////  1.1 - Load Data

# --------- Chose the periode

#years = [2020,2019,2018,2017,2016,2015,2014]
#years = [2020]
years = [2020,2019,2018,2017,2016]


# --------- Chose the area with departement code

#departements = ['75'] 
#departements = ['75','92','93','94','77','78','91','95'] 
departements = ['75','92','93','94']

with open("data_info.txt", 'w') as outfile:
    outfile.write("########################## Data Info ##########################\n")
    outfile.write("\nLocations: %s" % str(departements))
    outfile.write("\nPeriods: %s" % str(years))

	
data = dl.get_market_data(years = years,departements=departements,top_cities=None)

# ////////////  1.2 - Features engineering with special encoding

X_train,X_test,y_train,y_test = dp.feature_engineering(data)




####################################
# 		2 - Helper Functions
####################################

def model_selection(models,X,y,verbose=True):
    i=1
    model_score = []
    for name,model in models.items():
        scores = cross_val_score(model, X,y, cv=tscv)
        if verbose:
            print(f"{name} | score : {scores.sum()/5}")
        model_score.append((scores.sum()/5,model))
        i+=1
    best_model = sorted(model_score, reverse=True)[0][1]
    print('\n######## Best Model ########\n\t%s'%str(best_model))
    return best_model

	
####################################
# 		3 - Model Selection and Training
####################################

# ////////////  3.1 - Model preparation

ols = linear_model.LinearRegression()
ridge = linear_model.Ridge(alpha=.1)
lasso = linear_model.Lasso(alpha=0.1)
bayesian_ridge = linear_model.BayesianRidge()
svr = svm.SVR() # >== scale very badly
rf = RandomForestRegressor(max_depth=10,min_samples_leaf=10, random_state=0)
gbreg = GradientBoostingRegressor()

	
models = {'OLS regression' : ols,
          'Ridge regression' : ridge,
          'Lasso regression' : lasso,
          'Bayesian Ridge regression' : bayesian_ridge,
          'RandomForestRegressor' : rf,
          'Gradient Boosting Regressor':gbreg
          }
		  
# ////////////  3.2 - Features definition
features_augmented = ['surface','pieces','encodage_voie',
                      'terrain','id_local']

# ////////////  3.3 - Model Selection
model = model_selection(models,X_train[features_augmented],y_train)

# ////////////  3.4 - Best Model training
model.fit(X_train[features_augmented],y_train)

# ////////////  3.5 - Best Model save
#dump(model, open('model.pkl', 'wb'))

	
		
ploter.save_model_performances(X_train[features_augmented],y_train, model,file_name='performance_training',title='Performance on Training data')
ploter.save_model_performances(X_test[features_augmented],y_test, model,file_name='performance_test',title='Performance on Test data')
ploter.get_metrics(X_train[features_augmented],y_train,X_test[features_augmented],y_test, model)

####################################
# 		4 - Feature importance
####################################
try :
    feat_importances = pd.Series(model.feature_importances_, index=features_augmented)
except :    
    feat_importances = pd.Series(model.coef_, index=features_augmented)
feat_importances.plot(kind='barh',title='Feature Importance').get_figure().savefig("feature_importance.png")



####################################
# 		4 - Investigate errors 
####################################

# ////////////  4.1 - Investigate errors on Train Set
train_info = X_train[['Commune','surface','pieces']]
predictions_train = model.predict(X_train[features_augmented])

final_predictions_train = pd.concat([train_info.reset_index(),y_train.reset_index().drop(columns=['index']),pd.Series(predictions_train.reshape(-1))],axis=1)\
.drop(columns='index')


final_predictions_train.columns = ['commune','surface','pieces','target','pred']

final_predictions_train[['target','pred']] = np.exp(final_predictions_train[['target','pred']])

final_predictions_train['error'] = abs(final_predictions_train.target-final_predictions_train.pred)

# ////////////  4.2 - Mean error by City

errors_city = final_predictions_train.groupby(['commune'])\
                       .agg({'error':['min','mean','max','std']})\
                       .reset_index()

errors_city.columns=['commune','error_min','error_mean','error_max','std_error']
errors_city=errors_city.sort_values(by=['error_mean'])
errors_city['inv_error_mean']=errors_city['error_mean'].apply(lambda v : 1/v)
errors_city['inv_std_error']=errors_city['std_error'].apply(lambda v : 1/v)

# ////////////  4.2 - City with high mean errors
errors_city[['commune','error_mean']].tail(10)
#ploter.show_cloud(errors_city)
ploter.save_cloud(errors_city,val= 'error_mean',title='City with high mean errors',file_name='high_mean_errors')

# ////////////  4.3 - City with low mean errors
errors_city[['commune','error_mean']].head(0)
#ploter.show_cloud(errors_city,'inv_error_mean')
ploter.save_cloud(errors_city,val= 'inv_error_mean',title='City with low mean errors',file_name='low_mean_errors')

# ////////////  4.3 - City with low mean errors
#ploter.show_cloud(errors_city.dropna(),'std_error')

# ////////////  4.3 - City with high std errors
#ploter.show_cloud(errors_city.dropna(),'std_error')
ploter.save_cloud(errors_city,val= 'std_error',title='City with high std errors',file_name='high_std_errors')

# ////////////  4.3 - City with low std errors
#ploter.show_cloud(errors_city.dropna(),'inv_std_error')
ploter.save_cloud(errors_city,val= 'inv_std_error',title='City with low std errors',file_name='low_std_errors')
