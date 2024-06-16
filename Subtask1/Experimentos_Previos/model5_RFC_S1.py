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
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
    parser.add_argument('-v4', help='variable4')
    parser.add_argument('-v5', help='variable5')
    parser.add_argument('-v6', help='variable6')


    args = parser.parse_args()
    print('*'*30)
    print('Comienza procesamiento de datos')

    modelos=args.v4
    finetuned=str(args.v5)
    who=str(args.v6)


    if str(modelos)=='all':
        # Dataset con las características estilométricas
        train_stylometry=pd.read_csv('Train/train_stylometry_S1.csv')
        test_stylometry=pd.read_csv('Test/test_stylometry_S1.csv')
        train_stylometry=train_stylometry.iloc[:,2:15]
        test_stylometry=test_stylometry.iloc[:,2:15]
        columns = list(train_stylometry.columns)
        scaler = MinMaxScaler()
        scaler.fit(train_stylometry[columns])
        train_stylometry[columns] = scaler.transform(train_stylometry[columns])
        test_stylometry[columns] = scaler.transform(test_stylometry[columns])
        train_stylometry = train_stylometry.values 
        test_stylometry=test_stylometry.values
        
        if finetuned=='off':
            #Dataset LLM Bert
            train_bert=pd.read_csv('Train/train_subtask1bert-base-multilingual-cased.csv',header=None)
            test_bert=pd.read_csv('Test/test_subtask1bert-base-multilingual-cased.csv',header=None)
            train_bert=train_bert.values
            test_bert=test_bert.values
        
            # Dataset LLM Multilingual_e5 
            train_e5=pd.read_csv('Train/train_subtask1multilingual-e5-large.csv',header=None)
            test_e5=pd.read_csv('Test/test_subtask1multilingual-e5-large.csv',header=None)
            train_e5=train_e5.values
            test_e5=test_e5.values
        else:
            if who=='Andric':
                #Dataset LLM Bert
                train_bert=pd.read_csv('Train/train_subtask1bert-base-multilingual-cased-finetuned-autext24.csv',header=None)
                test_bert=pd.read_csv('Test/test_subtask1bert-base-multilingual-cased-finetuned-autext24.csv',header=None)
                train_bert=train_bert.values
                test_bert=test_bert.values
        
                # Dataset LLM Multilingual_e5 
                train_e5=pd.read_csv('Train/train_subtask1multilingual-e5-large-finetuned-autext24.csv', header=None)
                test_e5=pd.read_csv('Test/test_subtask1multilingual-e5-large-finetuned-autext24.csv',header=None)
                train_e5=train_e5.values
                test_e5=test_e5.values

            elif who=='Victor':
                #Dataset LLM Bert
                train_bert=pd.read_csv('Train/train_subtask1bert-base-multilingual-cased-finetuned-IberAuTexTification2024-7030-4epo-task1-v2.csv',header=None)
                test_bert=pd.read_csv('Test/test_subtask1bert-base-multilingual-cased-finetuned-IberAuTexTification2024-7030-4epo-task1-v2.csv',header=None)
                train_bert=train_bert.values
                test_bert=test_bert.values
        
                # Dataset LLM Multilingual_e5 
                train_e5=pd.read_csv('Train/train_subtask1multilingual-e5-large-finetuned-IberAuTexTification2024-7030-4epo-task1-v2.csv',header=None)
                test_e5=pd.read_csv('Test/test_subtask1multilingual-e5-large-finetuned-IberAuTexTification2024-7030-4epo-task1-v2.csv',header=None)
                train_e5=train_e5.values
                test_e5=test_e5.values
            



        # Dataset original
        train_data = pd.read_csv('Train/train_S1.csv')
        test_data = pd.read_csv('Test/test_S1.csv')
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
        X_train_cv=add_feature1(X_train_cv,train_stylometry)
        X_train_cv=add_feature1(X_train_cv,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
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
        # Calculamos más características adicionales
        num_digits_test= X_test_data.str.count('\d')
        num_stops_test = X_test_data.str.count('\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
    
        print('Termina procesamiento de datos')
        print('*'*30)
    elif str(modelos)=='sty+bert':
        # Dataset con las características estilométricas
        train_stylometry=pd.read_csv('Train/train_stylometry_S1.csv')
        test_stylometry=pd.read_csv('Test/test_stylometry_S1.csv')
        train_stylometry=train_stylometry.iloc[:,2:15]
        test_stylometry=test_stylometry.iloc[:,2:15]
        columns = list(train_stylometry.columns)
        scaler = MinMaxScaler()
        scaler.fit(train_stylometry[columns])
        train_stylometry[columns] = scaler.transform(train_stylometry[columns])
        test_stylometry[columns] = scaler.transform(test_stylometry[columns])
        train_stylometry = train_stylometry.values 
        test_stylometry=test_stylometry.values
        
        # Dataset LLM Bert
        train_bert=pd.read_csv('Train/train_subtask1bert-base-multilingual-cased.csv',header=None)
        test_bert=pd.read_csv('Test/test_subtask1bert-base-multilingual-cased.csv',header=None)
        train_bert=train_bert.values
        test_bert=test_bert.values
        # Dataset original
        train_data = pd.read_csv('Train/train_S1.csv')
        test_data = pd.read_csv('Test/test_S1.csv')
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
        X_train_cv=add_feature1(X_train_cv,train_stylometry)
        X_train_cv=add_feature1(X_train_cv,train_bert)
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
        # Calculamos más características adicionales
        num_digits_test= X_test_data.str.count('\d')
        num_stops_test = X_test_data.str.count('\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
        print('Termina procesamiento de datos')
        print('*'*30)

    elif str(modelos)=='sty+e5':
        # Dataset con las características estilométricas
        train_stylometry=pd.read_csv('Train/train_stylometry_S1.csv')
        test_stylometry=pd.read_csv('Test/test_stylometry_S1.csv')
        train_stylometry=train_stylometry.iloc[:,2:15]
        test_stylometry=test_stylometry.iloc[:,2:15]
        columns = list(train_stylometry.columns)
        scaler = MinMaxScaler()
        scaler.fit(train_stylometry[columns])
        train_stylometry[columns] = scaler.transform(train_stylometry[columns])
        test_stylometry[columns] = scaler.transform(test_stylometry[columns])
        train_stylometry = train_stylometry.values 
        test_stylometry=test_stylometry.values
    
        
        # Dataset LLM Multilingual_e5 
        train_e5=pd.read_csv('Train/train_subtask1multilingual-e5-large.csv',header=None)
        test_e5=pd.read_csv('Test/test_subtask1multilingual-e5-large.csv',header=None)
        train_e5=train_e5.values
        test_e5=test_e5.values
    
    
    
        # Dataset original
        train_data = pd.read_csv('Train/train_S1.csv')
        test_data = pd.read_csv('Test/test_S1.csv')
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
        X_train_cv=add_feature1(X_train_cv,train_stylometry)
        X_train_cv=add_feature1(X_train_cv,train_e5)
        # Calculamos más características adicionales 
        num_digits= X_train_data.str.count('\d')
        num_stops = X_train_data.str.count('\s')    
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_digits)
        X_train_cv = add_feature2(X_train_cv, num_stops)
    
        # PRUEBA 
        X_test_cv=cv.transform(X_test_data)
        X_test_cv=add_feature1(X_test_cv,test_stylometry)
        X_test_cv=add_feature1(X_test_cv,test_e5)
        # Calculamos más características adicionales
        num_digits_test= X_test_data.str.count('\d')
        num_stops_test = X_test_data.str.count('\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
    
        print('Termina procesamiento de datos')
        print('*'*30)
    elif str(modelos)=='e5+bert':
    
        # Dataset LLM Bert
        train_bert=pd.read_csv('Train/train_subtask1bert-base-multilingual-cased.csv',header=None)
        test_bert=pd.read_csv('Test/test_subtask1bert-base-multilingual-cased.csv',header=None)
        train_bert=train_bert.values
        test_bert=test_bert.values
        
        # Dataset LLM Multilingual_e5 
        train_e5=pd.read_csv('Train/train_subtask1multilingual-e5-large.csv',header=None)
        test_e5=pd.read_csv('Test/test_subtask1multilingual-e5-large.csv',header=None)
        train_e5=train_e5.values
        test_e5=test_e5.values
    
    
        # Dataset original
        train_data = pd.read_csv('Train/train_S1.csv')
        test_data = pd.read_csv('Test/test_S1.csv')
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
        X_train_cv=add_feature1(X_train_cv,train_bert)
        X_train_cv=add_feature1(X_train_cv,train_e5)
        # Calculamos más características adicionales 
        num_digits= X_train_data.str.count('\d')
        num_stops = X_train_data.str.count('\s')    
        # Y las agregamos a nuestros datos 
        X_train_cv = add_feature2(X_train_cv, num_digits)
        X_train_cv = add_feature2(X_train_cv, num_stops)
    
        # PRUEBA 
        X_test_cv=cv.transform(X_test_data)
        X_test_cv=add_feature1(X_test_cv,test_bert)
        X_test_cv=add_feature1(X_test_cv,test_e5)
        # Calculamos más características adicionales
        num_digits_test= X_test_data.str.count('\d')
        num_stops_test = X_test_data.str.count('\s')
        # Y las agregamos a nuestros datos
        X_test_cv = add_feature2(X_test_cv, num_digits_test)
        X_test_cv = add_feature2(X_test_cv, num_stops_test)
    
        print('Termina procesamiento de datos')
        print('*'*30)

    modelo_rfc = RandomForestClassifier(random_state=0)
    print('Comienza el entrenamiento')
    modelo_rfc.fit(X_train_cv, y_train_data)
    print('Finaliza entrenamiento')
    print('*'*30)
    predictions = modelo_rfc.predict(X_test_cv)
    score=f1_score(y_test_data,predictions, average='macro')
    confusion = confusion_matrix(y_test_data,predictions)
    with open('./RFC_S1.txt', 'a') as archivo:
        archivo.write(f'Parámetros: \n Datasets:{modelos} \n finetuning: {who} \n ngram_range: {ngram_range} \n analyzer: {analyzer} \n mind_df: {min_df} \n f1_score: {score} \n')

    

if __name__ == "__main__":
    main()