from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd


class BankNote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float


app=FastAPI()
pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message':'Hello world'}

@app.get('/{name}')
def get_name(name:str):
    return {'Welcome to banknote classification app':f'{name}'}

@app.post('/predict')
def predict_note(data:BankNote):
    data=data.dict()
    v=data['variance']
    s=data['skewness']
    c=data['curtosis']
    e=data['entropy']
    pred=classifier.predict([[v,s,c,e]])
    if(pred[0]>0.5):
        pred="Its a fake note"
    else:
        pred="Its a real banknote"
    return {'prediction':pred}

if __name__ == "__main__":
    uvicorn.run(app,host='127.0.0.1')