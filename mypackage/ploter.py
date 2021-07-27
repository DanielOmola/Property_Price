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

