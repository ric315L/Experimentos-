from transformers import AutoTokenizer, AutoModel, Trainer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tqdm.auto import tqdm
import argparse
import json
import evaluate

def data_to_vec(model, tokenizer, df_path, save_path, cls_pos=0):
  """ Genera y guarda un dataset con los embeddings generados por el modelo dado

  Parámetros
  ----------
  model: Model (HuggingFace)
    Modelo con el que se generaran los embeddings
  tokenizer: Tokenizer (HuggingFace)
    Tokenizador para generar los vectores de entrada
  df_path: str or Path
    Ruta del dataset en formato json
  save_path: str or Path
    Ruta para guardar el dataset de los embeddings
  cls_pos: int
    Posiciones el token CLS, por defecto es 0 pero algunos modelos lo tienen
    al final (-1)
  """

  with open(df_path) as f:
    data = pd.read_json(path_or_buf=df_path, lines=True)
    #data = pd.read_csv(filepath_or_buffer=df_path)
  output = open(save_path, 'wb')
  d = data['text']
  for text in tqdm(d):
    input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    if torch.cuda.is_available:
      input.to("cuda")
      model.to("cuda")
    o = model(**input).last_hidden_state[0,cls_pos,:].detach().to("cpu").numpy()
    np.savetxt(output, o[None], delimiter=',')
  output.close()
  
  
  
def main():
  parser = argparse.ArgumentParser(description='Descripción de tu script.')
  parser.add_argument('-i', '--input', required=True, help='Ruta del archivo de entrada.')
  parser.add_argument('-v1', '--v1', required=True, help='variable1')
  args = parser.parse_args()


  #models=[
   # "andricValdez/bert-base-multilingual-cased-finetuned-autext24",
   # "andricValdez/multilingual-e5-large-finetuned-autext24",
   # "vg055/xlm-roberta-base-finetuned-IberAuTexTification2024-7030-4epo-task1-v2"
   # ]

  models=["andricValdez/xlm-roberta-base-finetuned-autext24"]



  model_names = [m.split('/')[-1] for m in models]

  models_l = []
  tokenizers = []

  for m in models:
    tokenizer = AutoTokenizer.from_pretrained(m)
    model = AutoModel.from_pretrained(m)
    tokenizers.append(tokenizer)
    models_l.append(model)
  
  df_path = args.input
  save_path = "./" 
  data=int(args.v1) #Si data==0 es el conjunto de entrenamiento.  


  for i in range(len(models_l)):
    cls_pos = 0
    if data==0:
      data_to_vec(
        model=models_l[i],
        tokenizer=tokenizers[i],
        df_path=df_path,
        save_path=save_path + "train_subtask1" + model_names[i] + ".csv",
        cls_pos=cls_pos
      )
    else:
      data_to_vec(
        model=models_l[i],
        tokenizer=tokenizers[i],
        df_path=df_path,
        save_path=save_path + "test_subtask1" + model_names[i] + ".csv",
        cls_pos=cls_pos
      )

if __name__ == "__main__":
    main()

