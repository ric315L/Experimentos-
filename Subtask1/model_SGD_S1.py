# Importamos las bibliotecas a usar
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
import nltk
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.linear_model import SGDClassifier

# Función para añadir más de una característica
def add_feature1(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    # Se concatena la secuncia de caracteres del conjunto de train con un matriz dispersa
    return hstack([X, csr_matrix(feature_to_add)], 'csr')

# Función para añadir una característica a la vez
def add_feature2(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    # Se concatena la secuncia de caracteres del conjunto de train con un matriz dispersa
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
        

def main():
    print('*'*30)
    print('Comienza procesamiento de datos')
    train_stylometry=pd.read_csv('exp_train/stylometry_train_S1.csv')
    #test_stylometry=pd.read_csv('exp_test/stylometry_test_S1.csv')
    test_stylometry=pd.read_csv('Test/stylometry_test_S1.csv')
    train_stylometry = train_stylometry.values 
    test_stylometry=test_stylometry.values
    print('Stylometry listo!')
    
    
    #Dataset LLM Bert (Andric)
    train_bert=pd.read_csv('exp_train/train_subtask1bert-base-multilingual-cased-finetuned-autext24.csv',header=None)
    #test_bert=pd.read_csv('exp_test/test_subtask1bert-base-multilingual-cased-finetuned-autext24.csv',header=None)
    test_bert=pd.read_csv('Test/test_subtask1bert-base-multilingual-cased-finetuned-autext24.csv',header=None)
    train_bert=train_bert.values
    test_bert=test_bert.values
    print('LLM Bert listo!')
    
    # Dataset LLM Multilingual_e5 (Andric)
    train_e5=pd.read_csv('exp_train/train_subtask1multilingual-e5-large-finetuned-autext24.csv', header=None)
    #test_e5=pd.read_csv('exp_test/test_subtask1multilingual-e5-large-finetuned-autext24.csv',header=None)
    test_e5=pd.read_csv('Test/test_subtask1multilingual-e5-large-finetuned-autext24.csv',header=None)
    train_e5=train_e5.values
    test_e5=test_e5.values
    print('LLM E5 listo!')

    #Dataset roberta (Andric)
    train_roberta=pd.read_csv('exp_train/train_subtask1xlm-roberta-base-finetuned-autext24.csv',header=None)
    #test_roberta=pd.read_csv('exp_test/test_subtask1xlm-roberta-base-finetuned-autext24.csv',header=None)        
    test_roberta=pd.read_csv('Test/test_subtask1xlm-roberta-base-finetuned-autext24.csv',header=None)
    train_roberta=train_roberta.values
    test_roberta=test_roberta.values
    print('LLM Roberta listo!')


    # Dataset original
    train_data = pd.read_csv('exp_train/train_S1.csv')
    #test_data = pd.read_csv('exp_test/test_S1.csv')
    test_data = pd.read_json('Test/test_set_original.jsonl',lines=True)

    train_data['label'] = np.where(train_data['label']=='generated',1,0)

    #test_data['label'] = np.where(test_data['label']=='generated',1,0)

    X_train_data=train_data['text']
    y_train_data=train_data['label']
    
    X_test_data=test_data['text']
    #y_test_data=test_data['label']

    print(train_data.shape)
    print(test_data.shape)
    print('Datos originales, listo!')
    
        
        
    # Datos de entrenamiento 
    X_train_cv=add_feature1(train_stylometry,train_bert)
    X_train_cv=add_feature1(X_train_cv,train_e5)
    X_train_cv=add_feature1(X_train_cv,train_roberta)
    # Calculamos más características adicionales 
    num_digits= X_train_data.str.count('\d')
    num_stops = X_train_data.str.count('\s')    
    # Y las agregamos a nuestros datos 
    X_train_cv = add_feature2(X_train_cv, num_digits)
    X_train_cv = add_feature2(X_train_cv, num_stops)
    print('Datos de entrenamiento listos!')

    # Datos de prueba 
    X_test_cv=add_feature1(test_stylometry,test_bert)
    X_test_cv=add_feature1(X_test_cv,test_e5)
    X_test_cv=add_feature1(X_test_cv,test_roberta)
    # Calculamos más características adicionales
    num_digits_test= X_test_data.str.count('\d')
    num_stops_test = X_test_data.str.count('\s')
    # Y las agregamos a nuestros datos
    X_test_cv = add_feature2(X_test_cv, num_digits_test)
    X_test_cv = add_feature2(X_test_cv, num_stops_test)
    print('Datos de prueba listos!')

    print('Termina procesamiento de datos')
    print('*'*30)

    modelo_SGD = SGDClassifier(random_state=0)
    print('Comienza el entrenamiento')
    modelo_SGD.fit(X_train_cv, y_train_data)
    print('Finaliza entrenamiento')
    print('*'*30)
    predictions = modelo_SGD.predict(X_test_cv)
    #score=f1_score(y_test_data,predictions, average='macro')
    #print(score)
    test_data['label']=predictions
    test_data['label'] = np.where(test_data['label']==1,'generated','human')
    test_data=test_data[['id','label']]
    test_data.to_json('./resultados_S1.jsonl', orient='records', lines=True)


if __name__ == "__main__":
    main()
