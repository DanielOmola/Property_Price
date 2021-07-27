# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""

import plotly.graph_objects as go

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from wordcloud import WordCloud

    
def check_model_performances(X,Y, model,show=False):
    """
	####################################################################################
	#
	#  Plot Model predictions vs. target and print MSE and R2
    #  Inputs : X (features), Y (target), model (sci-kit learn model), show (Boolean)
	#
	####################################################################################
	"""
    #model.fit(X, Y)
    predictions = model.predict(X)
    
    predictions = predictions#.reshape(-1,1)
    
    # ######## Computes MSE #######    
    MSE = mean_squared_error(Y, predictions)
    print(f'\nMSE : {MSE}')
    
    # ######## Computes R2 ####### 
    R2 = r2_score(Y, predictions)
    print(f'R2 : {R2}')
    
    # ######## Plot Model predictions vs. target #######     
    if show:
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(y=Y,
                            mode='lines',
                            name='target'))
        fig.add_trace(go.Scatter(y=predictions
                                ,
                            mode='lines',
                            name='predictions'))
    
        fig.show()

def save_model_performances(X,Y, model,file_name='performance_training',title='Performance on training data'):
    """
	####################################################################################
	#
	#  Plot Model predictions vs. target and print MSE and R2
    #  Inputs : X (features), Y (target), model (sci-kit learn model), show (Boolean)
	#
	####################################################################################
	"""
    #model.fit(X, Y)
    predictions = model.predict(X)
    
    predictions = predictions#.reshape(-1,1)
    
    # ######## Computes MSE #######    
    MSE = mean_squared_error(Y, predictions)
    print(f'\nMSE : {MSE}')
    
    # ######## Computes R2 ####### 
    R2 = r2_score(Y, predictions)
    print(f'R2 : {R2}')
    title="%s MSE: %f, R2: %f"%(title,MSE,R2)
    # ######## Plot Model predictions vs. target #######     
    #if show:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=Y,
						mode='lines',
						name='target'))
    fig.add_trace(go.Scatter(y=predictions,
						mode='lines',
						name='predictions'))
    fig.update_layout(title=title)
    fig.write_image("%s.png"%file_name)
	
def get_metrics(X_train,Y_train,X_test,Y_test, model):
    """
	####################################################################################
	#
	#  Plot Model predictions vs. target and print MSE and R2
    #  Inputs : X (features), Y (target), model (sci-kit learn model), show (Boolean)
	#
	####################################################################################
	"""
    #model.fit(X, Y)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)   
    # ######## Computes MSE #######    
    MSE_train = mean_squared_error(Y_train, predictions_train)
    MSE_test = mean_squared_error(Y_test, predictions_test)
    
    # ######## Computes R2 ####### 
    R2_train = r2_score(Y_train, predictions_train)
    R2_test = r2_score(Y_test, predictions_test)

    with open("metrics.txt", 'w') as outfile:
        outfile.write("\n########################## Model ##########################\n\t%s" % str(model))
        outfile.write("\nTraining MSE: %2.4f%%" % MSE_train)
        outfile.write("\nTest MSE: %2.4f%%" % MSE_test)
        outfile.write("\nTraining R2: %2.4f%%" % R2_train)
        outfile.write("\nTest R2: %2.4f%%" % R2_test)
# ////////////////////////////////////////////////////////////////////
def compar_prediction_target(predictions,Y,show=False):
  from sklearn.metrics import mean_squared_error
  import plotly.graph_objects as go

  predictions = predictions#.reshape(-1,1)
  
  MSE = mean_squared_error(Y, predictions)
  print(f'MSE : {MSE}')

  from sklearn.metrics import r2_score
  R2 = r2_score(Y, predictions)
  print(f'R2 : {R2}')
  if show:
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(y=Y,
                        mode='lines',
                        name='target'))
    fig.add_trace(go.Scatter(y=predictions
                            ,
                        mode='lines',
                        name='predictions'))

    fig.show()




def show_cloud(data,val= 'error_mean'):
    E = data[['commune',val]]
    D = E.set_index('commune').T.to_dict('list')
    for k,v in D.items():
        D[k]=v[0]
    
    
    
    wordcloud = WordCloud(max_words  = 500, width = 1000, height = 500).generate_from_frequencies(D)
    
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)

def save_cloud(data,val= 'error_mean',title='Error Analysis',file_name='error_analysis'):
    E = data[['commune',val]]
    E.dropna(inplace=True)
    D = E.set_index('commune').T.to_dict('list')
    for k,v in D.items():
        D[k]=v[0]
    
    
    
    wordcloud = WordCloud(max_words  = 500, width = 1000, height = 500).generate_from_frequencies(D)
    
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.savefig('%s.png'%file_name)