# Importamos las bibliotecas a usar
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
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
    modelo_SGD = SGDClassifier(max_iter=1000, random_state=42)
    
    param_grid = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.01, 0.1, 1],
    'power_t': [0.25, 0.5]}

    grid_search = GridSearchCV(modelo_SGD, param_grid, cv=5, scoring='f1_macro', n_jobs=1)
    print('Comienza el entrenamiento')
    grid_search.fit(X_train_cv, y_train_data)
    print('Finaliza')
    print('*'*100)


    # Abrir un archivo en modo escritura
    with open('GS_model3_S1.txt', 'a') as file:
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
