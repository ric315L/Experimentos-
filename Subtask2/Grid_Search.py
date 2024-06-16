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
from sklearn.linear_model import LogisticRegression
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


    # Dataset con las características estilométricas
    train_stylometry=pd.read_csv('exp_train/stylometry_train_S2.csv')
    test_stylometry=pd.read_csv('exp_test/stylometry_test_S2.csv')
    #train_stylometry=train_stylometry.iloc[:,2:15]
    #test_stylometry=test_stylometry.iloc[:,2:15]
    #columns = list(train_stylometry.columns)
    #scaler = MinMaxScaler()
    #scaler.fit(train_stylometry[columns])
    #train_stylometry[columns] = scaler.transform(train_stylometry[columns])
    #test_stylometry[columns] = scaler.transform(test_stylometry[columns])
    train_stylometry = train_stylometry.values 
    test_stylometry=test_stylometry.values
    
    #Dataset LLM Bert
    train_bert=pd.read_csv('exp_train/train_subtask2bert-base-multilingual-cased-finetuned-autext24-subtask2.csv',header=None)
    test_bert=pd.read_csv('exp_test/test_subtask2bert-base-multilingual-cased-finetuned-autext24-subtask2.csv',header=None)
    train_bert=train_bert.values
    test_bert=test_bert.values
    # Dataset LLM Multilingual_e5 
    train_e5=pd.read_csv('exp_train/train_subtask2multilingual-e5-large-finetuned-autext24-subtask2.csv', header=None)
    test_e5=pd.read_csv('exp_test/test_subtask2multilingual-e5-large-finetuned-autext24-subtask2.csv',header=None)
    train_e5=train_e5.values
    test_e5=test_e5.values
        
    #Dataset roberta 
    train_roberta=pd.read_csv('exp_train/train_subtask2xlm-roberta-base-finetuned-autext24-subtask2.csv',header=None)
    test_roberta=pd.read_csv('exp_test/test_subtask2xlm-roberta-base-finetuned-autext24-subtask2.csv',header=None)
    train_roberta=train_roberta.values
    test_roberta=test_roberta.values

    # Dataset original
    train_data = pd.read_csv('exp_train/train_S2.csv')
    test_data = pd.read_csv('exp_test/test_S2.csv')
    etiquetas = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, clase in enumerate(etiquetas):
        train_data['label'] = np.where(train_data['label'] == clase, i, train_data['label'])

    for i, clase in enumerate(etiquetas):
        test_data['label'] = np.where(test_data['label'] == clase, i, test_data['label'])

    X_train_data=train_data['text']
    y_train_data=train_data['label']
    X_test_data=test_data['text']
    y_test_data=test_data['label']
    y_train_data = y_train_data.astype(int)
    y_test_data = y_test_data.astype(int)
        
    ngram_range = (3,5)
    analyzer='char'
    min_df=2500
        
    cv = CountVectorizer(ngram_range=ngram_range,analyzer=analyzer,min_df=min_df)
        
    # ENTRENAMIENTO 
    X_train_cv = cv.fit_transform(X_train_data)
    X_train_cv=add_feature1(X_train_cv,train_stylometry)
    X_train_cv=add_feature1(X_train_cv,train_bert)
    X_train_cv=add_feature1(X_train_cv,train_e5)
    X_train_cv=add_feature1(X_train_cv,train_roberta)
    # Calculamos más características adicionales 
    num_digits= X_train_data.str.count('\d')
    num_stops = X_train_data.str.count('\s')    
    # Y las agregamos a nuestros datos 
    X_train_cv = add_feature2(X_train_cv, num_digits)
    X_train_cv = add_feature2(X_train_cv, num_stops)
    
    # PRUEBA 
    X_test_cv=cv.transform(X_test_data)
    X_test_cv=add_feature1(X_test_cv,test_stylometry)
    X_test_cv=add_feature1(X_test_cv,test_bert)
    X_test_cv=add_feature1(X_test_cv,test_e5)
    X_test_cv=add_feature1(X_test_cv,test_roberta)
    # Calculamos más características adicionales
    num_digits_test= X_test_data.str.count('\d')
    num_stops_test = X_test_data.str.count('\s')
    # Y las agregamos a nuestros datos
    X_test_cv = add_feature2(X_test_cv, num_digits_test)
    X_test_cv = add_feature2(X_test_cv, num_stops_test)
    
    print('Termina procesamiento de datos')
    print('*'*30)


    modelo_SGD = SGDClassifier(random_state=0,max_iter=1000)

    # 3. Definir los hiperparámetros a explorar
    param_grid = {
        'loss': ['hinge', 'log', 'modified_huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.01, 0.1]}

    grid_search = GridSearchCV(modelo_SGD, param_grid, cv=5, scoring='f1_macro', n_jobs=1)
    print('Comienza el entrenamiento')
    grid_search.fit(X_train_cv, y_train_data)
    print('Finaliza')
    print('*'*100)

    # Abrir un archivo en modo escritura
    with open('GS.txt', 'a') as file:
        # Escribir los mejores parámetros
        file.write(f"Mejores parámetros: {grid_search.best_params_}\n")
        # Escribir la mejor puntuación de f1_macro
        file.write(f"Mejor puntuación de f1_macro: {grid_search.best_score_}\n")
        
        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        # Calcular la precisión en el conjunto de prueba
        test_accuracy = best_model.score(X_test_cv, y_test_data)
        # Escribir la precisión en el conjunto de prueba
        file.write(f"F1_macro en el conjunto de prueba: {test_accuracy}\n")

if __name__ == "__main__":
    main()
