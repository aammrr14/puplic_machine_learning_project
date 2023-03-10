# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pickle
import streamlit as st
import os

#loading the models 
path_list=os.listdir("/home/aammrr3/Downloads/deployment")
path_list.remove('web app.py')
path_list.remove('espilon.ai project (Compressive Strength prediction for concrete).ipynb')
model=[]
for model_filepath in path_list:
    with open(model_filepath,"rb") as f:
        model.append(pickle.load(f))
        
def make_pred(data):
    
    data=np.asarray(data).reshape(1, -1)
    
    linear_pred=model[0].predict(data)
    XG_pred=model[4].predict(data)
    GB_pred=model[2].predict(data)
    RF_pred=model[1].predict(data)
    CB_pred=model[3].predict(data)
    
    
    return linear_pred,XG_pred,GB_pred,RF_pred,CB_pred        

def main():
    
    #cereating the title 
    st.title("compresive strength prediction for cconcrete")
    
    


    #creating input from the user
    CementComponent=st.number_input('Cement')
    WaterComponent=st.number_input('Water')
    SuperplasticizerComponent=st.number_input('Super-plasticizer')
    CoarseAggregateComponent=st.number_input('Coarse Aggregate')
    FineAggregateComponent=st.number_input('Fine Aggregate')
    AgeInDays=st.number_input('Age')
    data=[CementComponent,WaterComponent,SuperplasticizerComponent,CoarseAggregateComponent,FineAggregateComponent,AgeInDays]
    
    
    compressive_strength=0
    
    #creating buttons for predecting on deffirent models
    linear_button=st.button("prediction based on RIDGE model ")
    XG_button=st.button("prediction based on XGBOOST model ")
    GB_button=st.button("prediction based on GRADIENT BOOSTING model ")
    RF_button=st.button("prediction based on RANDOM FOREST model ")
    CB_button=st.button("prediction based on CAT BOOST model ")
    
    if linear_button:
        linear_pred,XG_pred,GB_pred,RF_pred,CB_pred=make_pred(data)
        compressive_strength=linear_pred
    elif  XG_button :
        linear_pred,XG_pred,GB_pred,RF_pred,CB_pred=make_pred(data)
        compressive_strength=XG_pred
    elif GB_button:
        linear_pred,XG_pred,GB_pred,RF_pred,CB_pred=make_pred(data)
        compressive_strength=GB_pred
    elif RF_button:
        linear_pred,XG_pred,GB_pred,RF_pred,CB_pred=make_pred(data)
        compressive_strength=RF_pred
    elif  CB_button:  
        linear_pred,XG_pred,GB_pred,RF_pred,CB_pred=make_pred(data)
        compressive_strength=CB_pred
    
    
    st.success(str(compressive_strength))
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    