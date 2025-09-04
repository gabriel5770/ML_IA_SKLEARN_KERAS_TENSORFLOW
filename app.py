from zlib import crc32
import os 
import tarfile
import urllib
import urllib.request

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-m12/master/'
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'  

# def fetch_housing_data(housing_url = HOUSING_URL,housing_path=HOUSING_PATH):
#     tgz_path = os.makedirs(housing_path,exist_ok=True)
#     urllib.request.urlretrieve(housing_url, tgz_path)
#     housing_tgz = tarfile.open(tgz_path)
#     housing_tgz.extractall(path=housing_path)
#     housing_tgz.close()

# fetch_housing_data()

import pandas as pd
  
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#mostra as informações iniciais do data base
#print(housing.head())

#mostra os tipos das colunas, quantidades , memória usada e etc..
#print(housing.info())

#mostra os tipos de dados de uma respectiva coluna , no caso da coluna categória ocean_proximity
#print(housing["ocean_proximity"].value_counts())

#mas pode ser de uma não categória também
#print(housing["total_rooms"].value_counts())

#mostra a descrição resumida de cada campo
#print(housing.describe())

import matplotlib as plt


housing.hist(bins=50,figsize=(20,15))
#plt.show()

import numpy as np

#Essa função serve para dividir o dataset em treino e teste de forma fixa, 
# garantindo que os mesmos registros sempre fiquem no mesmo conjunto, 
# mesmo se você rodar o código várias vezes ou atualizar os dados. 

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data,test_ratio,id_column): 
    ids = data[id_column] 
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))        
    return data.loc[~in_test_set], data.loc[in_test_set]

    #housing_with_id = housing.reset_index()
    #train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")

#print(train_set)
housing_with_id = housing.reset_index()
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")


##substitui a função manual da linha 55 À 61
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")  # ou "Qt5Agg"
import matplotlib.pyplot as plt


train_set , test_set = train_test_split(housing,test_size=0.2,random_state=42)

#print(test_set)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins =[0.,1.5,3.0,4.5,6.,np.inf],
                               labels=[1,2,3,4,5])

housing["income_cat"].hist()

plt.show()

