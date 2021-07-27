# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from pickle import dump

import sqlite3

def get_historical_price(r,prices):
    """
    #######################################################
    #
    #  Get historical prices for a transaction
    #
    #######################################################
    """
    p = prices[(prices['Date mutation']<r[0])\
              & (prices['Date mutation']>=r[1])\
              & (prices['Commune']==r[2])\
              & (prices['pieces']==r[4])]
    try :
        return p.prix_m2.median()
    except :
        return np.nan
    
def add_historical_prices(data):
    """
    #######################################################
    #
    #  Get historical prices for every transaction
    #
    #######################################################
    """
    prices = pd.read_csv('./data/prices.csv')
    prices.loc[:,'Date mutation']=pd.to_datetime(prices.loc[:,'Date mutation'])
    data['historical_price'] = data[['Date mutation','d_','Commune','adresse','pieces']]\
    .apply(lambda r : get_historical_price(r,prices),axis=1)
    data=data[~data['historical_price'].isna()]
    data=data[data['historical_price']!=0]
    #data.to_csv('data_clean.csv',index=False)
    data.to_csv('data_clean.csv.zip',index=False,compression="zip")
    return data    
    

def feature_engineering(data):
    """
    #######################################################
    #
    #  Get historical prices for every transaction
    #
    #######################################################
    """
    # ############## Target log transformation
    data['prix_m2'] = np.log(data['prix_m2'])
    #data['historical_price'] = np.log(data['historical_price'])
    data=data.sort_values(by=['Date mutation'])
    
    # ############## Create new categories/columns    
    data['ville_type_bien']=data[['Commune','Type local']].apply(lambda r :'%s %s'%(r[0],r[1]),axis=1)
    data['ville_voie']=data[['Commune','Voie']].apply(lambda r :'%s %s'%(r[0],r[1]),axis=1)
    data['ville_type_voie_voie']=data[['Commune','Type de voie','Voie']].apply(lambda r :'%s %s %s'%(r[0],r[1],r[2]),axis=1)
    data['ville_piece']=data[['Commune','pieces']].apply(lambda r :'%s %s'%(r[0],r[1]),axis=1)
    #data['departement_type_bien']=data[['Code departement','Type local']].apply(lambda r :'%s %s'%(r[0],r[1]),axis=1)
    data['departement_type_bien']=data[['dep','Type local']].apply(lambda r :'%s %s'%(r[0],r[1]),axis=1)
    data['ville_type_voie']=data[['Commune','Type de voie']].apply(lambda r :'%s %s'%(r[0],r[1]),axis=1)
    
    
    # ############## Create X and y before train/test split
    X = data.drop(columns=['prix_m2','Valeur fonciere'])
    y=data.prix_m2
    
    # ############## Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=False)
    #train_info = X_train[['Commune','surface','pieces']]
    #test_info = X_test[['Commune','surface','pieces']]    
    
    # ############## Categorical encoding based on y_train
    categorical = ['Code postal','dep','Commune','Code type local',
               'Type local','Voie','Type de voie','Code commune',
               'Nombre de lots','ville_type_bien','ville_voie','ville_piece',
              'departement_type_bien','ville_type_voie','ville_type_voie_voie']
    numerical = [c for c in data.columns[1:] if c not in categorical ]
    
    for c in data.columns[1:]:
        #print(c)
        if  not (c in ['Date mutation','d_','adresse','key']):
            if c in categorical :
                data[c]=data[c].astype('category')
            elif c in numerical  :
                data[c]=data[c].astype('float')
    #return data 

    # ################# ville encoding
    X_y_train = pd.concat([X_train.reset_index(),y_train.reset_index()],axis=1)
    X_y_train
    
    encoding = X_y_train.groupby(['Commune'])['prix_m2'].mean().sort_values(ascending=False).reset_index()
    encoding.columns=['Commune','encodage_ville']
    
    X_train = X_train.merge(encoding,how='left', left_on ='Commune', right_on ='Commune')
    X_test = X_test.merge(encoding,how='left', left_on ='Commune', right_on ='Commune')

    # ################# ville_type_bien encoding
    encoding = X_y_train.groupby(['ville_type_bien'])['prix_m2'].mean().sort_values(ascending=False).reset_index()
    encoding.columns=['ville_type_bien','encodage_ville_type_bien']
    
    X_train = X_train.merge(encoding,how='left', left_on ='ville_type_bien', right_on ='ville_type_bien')
    X_test = X_test.merge(encoding,how='left', left_on ='ville_type_bien', right_on ='ville_type_bien')

    # ################# departement_type_local encoding
    encoding = X_y_train.groupby(['departement_type_bien'])['prix_m2'].mean().sort_values(ascending=False).reset_index()
    encoding.columns=['departement_type_bien','encodage_departement']
    X_train = X_train.merge(encoding,how='left', left_on ='departement_type_bien', right_on ='departement_type_bien')
    X_test = X_test.merge(encoding,how='left', left_on ='departement_type_bien', right_on ='departement_type_bien')
    

    # ################# ville_voie encoding
    encoding = X_y_train.groupby(['ville_type_voie_voie'])['prix_m2'].mean().sort_values(ascending=False).reset_index()
    encoding.columns=['ville_voie','encodage_voie']
    
    X_train = X_train.merge(encoding,how='left', left_on ='ville_type_voie_voie', right_on ='ville_voie')
    X_test = X_test.merge(encoding,how='left', left_on ='ville_type_voie_voie', right_on ='ville_voie')
    
    # ################# ville_piece encoding
    encoding = X_y_train.groupby(['ville_piece'])['prix_m2'].mean().sort_values(ascending=False).reset_index()
    encoding.columns=['ville_piece','encodage_piece']
    
    X_train = X_train.merge(encoding,how='left', left_on ='ville_piece', right_on ='ville_piece')
    X_test = X_test.merge(encoding,how='left', left_on ='ville_piece', right_on ='ville_piece')   
    
    # ################# ville_type_voie encoding
    encoding = X_y_train.groupby(['ville_type_voie'])['prix_m2'].mean().sort_values(ascending=False).reset_index()
    encoding.columns=['ville_type_voie','encodage_type_voie']
    
    X_train = X_train.merge(encoding,how='left', left_on ='ville_type_voie', right_on ='ville_type_voie')
    X_test = X_test.merge(encoding,how='left', left_on ='ville_type_voie', right_on ='ville_type_voie')

    # ################# Clean test data
    test = pd.concat([X_test.reset_index(),y_test.reset_index()],axis=1)
    test=test[~test['encodage_voie'].isna()]
    test=test[~test['encodage_piece'].isna()]
    test=test[~test['encodage_type_voie'].isna()]
    test=test[~test['encodage_departement'].isna()]
    X_test =test.drop(columns=['prix_m2'])
    y_test = test.prix_m2

    # ################# Scaling
    
    for c in X_train.columns:
        try:
            scaler = MinMaxScaler() # StandardScaler()
            scaler = scaler.fit(X_train.loc[:,[c]])
            X_train.loc[:,c] = scaler.transform(X_train.loc[:,[c]])
            X_test.loc[:,c] = scaler.transform(X_test.loc[:,[c]])
        except:
            pass
    #dump(scaler, open('scaler.pkl', 'wb'))
  
    X_train.to_csv('./data/X_train.csv',index=False)
    X_test.to_csv('./data/X_test.csv',index=False)
    y_train.to_csv('./data/y_train.csv',index=False)
    y_test.to_csv('./data/y_test.csv',index=False)
    create_db_adresses()
    load_db_adresses()
    return X_train,X_test,y_train,y_test


def plot_NA(df):
  """
  ####################################################################################
  #
  #				Check and plot NaN on selected column
  #
  ####################################################################################
  """
  ################# Helper Functions ####################
  def check_na_col(df):
	  cl = df.columns[df.isna().any()].tolist()
	  return cl

  def check_number_na(df,na_col):
	  r = df[na_col].isna().sum()
	  return r
		
  def check_percent_na(df,na_col):
	  return (abs(df[na_col].count()-df[na_col].count().max()))/df[na_col].shape[0]*100
	#######################################################
	
  na_col = check_na_col(df)
	
  NA=check_number_na(df,na_col)
  NA=check_percent_na(df,na_col)
	
  fig = px.bar(NA)
  fig.show()
  


def create_db_adresses():
    bdd = sqlite3.connect('./data/base_adresses.db')
    requeteur = bdd.cursor()
    sql_sup_table_n = "DROP TABLE IF EXISTS adresse;"
    sql_table_adresse = """
            CREATE TABLE adresse 
            (commune VARCHAR(30) NOT NULL,
             type_voie VARCHAR(30) NOT NULL,
             voie VARCHAR(30)  NOT NULL,
             PRIMARY KEY (commune, type_voie, voie));"""

    requetes =[sql_sup_table_n,sql_table_adresse]
    for sql in requetes:
        requeteur.execute(sql)
    
    bdd.commit()

def load_db_adresses():
    bdd = sqlite3.connect('./data/base_adresses.db')
    requeteur = bdd.cursor()
    
    data = pd.read_csv('./data/X_train.csv',usecols=['Commune','Type de voie','Voie'])
    data=data.append(pd.read_csv('./data/X_test.csv',usecols=['Commune','Type de voie','Voie']))
    data=data.drop_duplicates()
    data['info'] = data[['Commune','Type de voie','Voie']].apply(lambda r : tuple(r),axis=1)
    info = data['info'].to_list()
    q = ",".join(map(lambda t : str(t),info))
    sql = "INSERT INTO adresse (commune, type_voie, voie) VALUES %s;"%q
    requeteur.execute(sql)
    bdd.commit()

def get_commune():
    bdd = sqlite3.connect('./data/base_adresses.db')
    requeteur = bdd.cursor()
    sql = "select Commune from adresse;"
    requeteur.execute(sql)
    resultat = requeteur.fetchall()
    bdd.close()
    print(resultat)

def get_voie(commune):
    bdd = sqlite3.connect('./data/base_adresses.db')
    requeteur = bdd.cursor()
    sql = "SELECT DISTINCT Voie from adresse WHERE Commune='%s';"%commune
    print(sql)
    requeteur.execute(sql)
    resultat = requeteur.fetchall()
    bdd.close()
    print(resultat)

def get_type_voie(commune):
    bdd = sqlite3.connect('./data/base_adresses.db')
    requeteur = bdd.cursor()
    sql = """SELECT DISTINCT type_voie from adresse WHERE Commune='%s';"""%commune
    print(sql)
    requeteur.execute(sql)
    resultat = requeteur.fetchall()
    bdd.close()
    print(resultat)

def get_adresse(commune):
    bdd = sqlite3.connect('./data/base_adresses.db')
    requeteur = bdd.cursor()
    sql = """SELECT DISTINCT type_voie, Voie from adresse WHERE Commune='%s';"""%commune
    requeteur.execute(sql)
    resultat = requeteur.fetchall()
    bdd.close()
    return resultat