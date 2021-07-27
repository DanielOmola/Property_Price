# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:45:01 2021

@author: daniel omola
"""


import numpy as np
import pandas as pd
import dateutil.relativedelta

def get_dropbox_link(year):
    DVI = ["https://www.dropbox.com/s/rmuej0caz03hqlk/valeursfoncieres-2020.zip?dl=1",
    "https://www.dropbox.com/s/i0lgj7gc70h33sr/valeursfoncieres-2019.zip?dl=1",
    "https://www.dropbox.com/s/sbcoqaqja94qlm4/valeursfoncieres-2018.zip?dl=1",
    "https://www.dropbox.com/s/wydzvm25xu4ubtl/valeursfoncieres-2017.zip?dl=1",
    "https://www.dropbox.com/s/j0jrq383g6q5qwh/valeursfoncieres-2016.zip?dl=1",
    "https://www.dropbox.com/s/6cm62rj7pwoz2er/valeursfoncieres-2015.zip?dl=1",
    "https://www.dropbox.com/s/2n46q8fa4qosyhs/valeursfoncieres-2014.zip?dl=1"]
    i = list(map(lambda link : str(year) in link,DVI)).index(True)  
    return DVI[i]


def get_market_data_by_year(year=2020, departements = None):
    """
    #######################################################
    #
    #   Loads, clean and preprocess data for a single year
    #
    #######################################################
    """
    
    # /////////////////// NESTED FUNCTIONS /////////////////
    
    def remove_outliers(data, col):
        """
        #######################################################
        #   Helper function for outliers removal
        #######################################################
        """
        def q1(x):
            q = np.percentile(x,25)
            return q

        def q3(x):
            q = np.percentile(x,75)
            return q

        IR = q3(data[col])-q1(data[col])
        outliers_l = q1(data[col])-1.5*IR
        outliers_u = q3(data[col])+1.5*IR

        data = data[(data[col]>=outliers_l) & (data[col]<=outliers_u)]

        return data
    
    def built_key(r):
        """
        #######################################################
        #   Helper function that build key for a transaction
        #######################################################
        """
        row = [str(v) for v in r]
        return "".join(row)
    
    # /////////////////////////////////////////////////////////
    
        
    # ############ keep predifned columns
    features_initial = ['Commune','Type local','Type de voie', 'Surface reelle bati',
                    'Nombre pieces principales','Surface terrain','Valeur fonciere',
                   'Nature mutation','Voie','Date mutation','Code postal', 'Code departement']
    
    # ############ read txt file
    #data = pd.read_csv('./data/valeursfoncieres-%d.txt'%year,sep='|',usecols=features_initial)
    dvf = get_dropbox_link(year)
    print(dvf)
    data = pd.read_csv(dvf,sep='|',usecols=features_initial,compression='zip')    
    print(f"valeursfoncieres-{year} : \n\t-initial shape {str(data.shape)}")
    
    
    
    occurrence_commune = data[['Commune']].value_counts().reset_index()
    occurrence_commune.columns= ['Commune','occurence']
    
    data = data.merge(occurrence_commune,how='left',on='Commune')
    #data['dep'] = data['Code postal'].apply(lambda cp : str(cp)[:2])
    data['dep'] = data['Code departement'].astype(str)
    
    if departements:
        data= data[data['dep'] .isin(departements)]
    #data = data[data.occurence>252]
    #print(data.columns)
    #print(len(data['Commune'].unique()))
          

    # ############ filter type of sale : 'Maison', 'Appartement'
    data = data[data['Type local'].isin(['Maison', 'Appartement'])]
    
    # ############ keep only sales
    data = data[data['Nature mutation']=='Vente']
    data = data.drop(columns=['Nature mutation'])
    
    # ############ remove NaN in 'Valeur fonciere'
    data=data[~data['Valeur fonciere'].isna()]
    
    # ############ removeremove comma in valeur foncière
    data['Valeur fonciere'] = data['Valeur fonciere'].apply(lambda v:float(v.split(',')[0]))
    #data = data.fillna(0)
    
    # ############ remove NaN in 'Type de voie' and 'Voie'
    data=data[~data['Type de voie'].isna()]
    data=data[~data['Voie'].isna()]

    # ############ keep house with 1 to 10 rooms
    data=data[(data['Nombre pieces principales']>0) & (data['Nombre pieces principales']<10)]
    
    # ############ compute price by square meter
    #data['prix_m2']=data['Valeur fonciere']/data['Surface reelle bati']
    

    
    # ############ Set 'Date Mutation' to datetime
    data.loc[:,'Date mutation']=pd.to_datetime(data.loc[:,'Date mutation'])

    # ############ fill nan with 0 for 'Surface terrain'
    data['Surface terrain']=data['Surface terrain'].apply(lambda v : 0 if str(v)=='nan' else v)
    
    # ############ create an identifier for each transaction
    data['key'] = data[['Commune','Type de voie','Valeur fonciere',
                   'Voie','Date mutation']].apply(lambda r : built_key(list(r)),axis=1)

    # ############ deal with duplicated transaction 
    grp = data\
    .groupby('key').agg({'Surface reelle bati':['sum'],
                        'Nombre pieces principales':['mean'],
                        'Surface terrain':['mean']}).reset_index()
    grp.columns=['key','surface','pieces','terrain']
    grp.pieces = grp.pieces.apply(lambda v : int(round(v)))

    data= data.merge(grp,how='left',on='key')\
    .drop(columns=[ 'Nombre pieces principales','Surface reelle bati','Surface terrain'])
    
    # ############ compute price by square meter
    data['prix_m2']=data['Valeur fonciere']/data['surface']
    
    # ############ keep house with price/m2 between 800 and 30000
    data=data[(data.prix_m2>800) & (data.prix_m2<30000)]
    
    # ############ drop duplicates
    data.drop_duplicates(inplace=True)
    
    # ############ add a new date (N month before the transaction date)
    data['d_']=data['Date mutation'].apply(lambda d :\
                                            d - dateutil.relativedelta.relativedelta(months=12) )
    data['id_local'] = data['Type local'].apply(lambda v : 0 if v=='Appartement' else 1)
    data = remove_outliers(data, 'prix_m2')
    data['adresse']=data[['Commune','Type de voie','Voie']].apply(lambda r : "".join(list(r)),axis=1)
    data = data.drop(columns=['occurence'])
    print(f"\t-final shape {str(data.shape)}")
    #print(data.shape)
    
    return data

def get_market_data(years = [2020],departements = ['92','95'],top_cities=10):
    """
    #######################################################
    #
    #  Loads, clean and preprocess data fora list of years
    #
    #######################################################
    """
    data = pd.DataFrame()
    for y in years:
        df = get_market_data_by_year(year=y,departements = departements)
        data=data.append(df,ignore_index=True)
    if top_cities:
        group_cities = data.groupby(['Commune']).Commune.count().sort_values(ascending=False)
        top = group_cities[:top_cities].index.to_list()
        data = data[data['Commune'].isin(top)]

    print(f"\n=> Final Data Frame shape: {str(data.shape)}")
    get_prices(data)
    return data

def get_prices(data):
    """
    #######################################################
    #
    #  Builds table of historical prices
    #  with info on city, adress and rooms 
    #
    #######################################################
    """
    prices  = data[['Date mutation','Commune','prix_m2','adresse','pieces']]\
    .groupby(['Date mutation','Commune','adresse','pieces'])\
    .agg({'prix_m2':'mean'}).reset_index()
    prices.to_csv('./data/prices.csv')
    #return prices