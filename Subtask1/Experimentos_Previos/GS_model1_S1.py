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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import uniform
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
    parser = argparse.ArgumentParser(description='Descripción de tu script.')
    #parser.add_argument('-v1', '--v1', help='variable1')
    parser.add_argument('-v1', help='variable1')
    parser.add_argument('-v2', help='variable2')
    parser.add_argument('-v3', help='variable3')


    args = parser.parse_args()
    print('*'*30)
    print('Comienza procesamiento de datos')


    train_data = pd.read_csv('./Train/train_S1.csv')
    test_data = pd.read_csv('./Test/test_S1.csv')
    train_data['label'] = np.where(train_data['label']=='generated',1,0)
    test_data['label'] = np.where(test_data['label']=='generated',1,0)
    X_train_data=train_data['text']
    y_train_data=train_data['label']
    X_test_data=test_data['text']
    y_test_data=test_data['label']
    
    ngram_range = eval(args.v1)
    analyzer=str(args.v2)
    min_df=int(args.v3)
    
    cv = CountVectorizer(ngram_range=ngram_range,analyzer=analyzer,min_df=min_df)
    
    # ENTRENAMIENTO 
    X_train_cv = cv.fit_transform(X_train_data)
    # Calculamos más características adicionales 
    num_digits= X_train_data.str.count('\d')
    num_stops = X_train_data.str.count('\s')    
    # Y las agregamos a nuestros datos 
    X_train_cv = add_feature2(X_train_cv, num_digits)
    X_train_cv = add_feature2(X_train_cv, num_stops)

    # PRUEBA 
    X_test_cv=cv.transform(X_test_data)
    # Calculamos más características adicionales
    num_digits_test= X_test_data.str.count('\d')
    num_stops_test = X_test_data.str.count('\s')
    # Y las agregamos a nuestros datos
    X_test_cv = add_feature2(X_test_cv, num_digits_test)
    X_test_cv = add_feature2(X_test_cv, num_stops_test)

    print('Termina procesamiento de datos')
    print('*'*30)


    print('Definimos el modelo')
    modelo_xgb = xgb.XGBClassifier()
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]}
    

    grid_search = GridSearchCV(modelo_xgb, param_grid, cv=5, scoring='f1_macro', n_jobs=1)
    print('Comienza el entrenamiento')
    grid_search.fit(X_train_cv, y_train_data)
    print('Finaliza')
    print('*'*100)



    # Abrir un archivo en modo escritura
    with open('GS_model1_S1.txt', 'a') as file:
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
