# Import necessary packages
import pandas as pd
import numpy as np
import torch
from torch import nn
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Setting up the Streamlit page
st.title(":red[MENTAL HEALTH SURVEY]")
with st.sidebar:
    sel = option_menu('Menu',["About","Predict Depression Status"])
    
if sel=='About':
    st.write("Depression is one of the major problems that people face today. There are several factors that cause depression - work-related stress, relationships, etc. In this app, the user can get the depression status, based on the parameters such as demographic information, lifestyle choices, and medical history. A deep learning model is used to make the prediction.")

# Load the updated CSV file
dep = pd.read_csv('Mental health data updated.csv')
DEGREE = tuple(dep.Degree.unique())
SUICIDAL_THOUGHTS = tuple(dep['Have you ever had suicidal thoughts ?'].unique())
MEDICAL_ILLNESS = tuple(dep['Family History of Mental Illness'].unique())
GENDER = tuple(dep.Gender.unique())
CITY = tuple(dep.City.unique())
PROF_OR_STUD = tuple(dep['Working Professional or Student'].unique())
PROFESSION = tuple(dep.Profession.unique())
SLEEP = tuple(dep['Sleep Duration'].unique())
DIETARY = tuple(dep['Dietary Habits'].unique())

# Model Architecture
class DeepMedical(nn.Module):
    def __init__(self,input_size,output_size):
        super(DeepMedical,self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,output_size),
            nn.Sigmoid())
        
    def forward(self,X):
        X = self.fnn(X)
        return X

# Load the model
model_mental = torch.load('medical_model.pth')

# Load the scaler
mental_sc = joblib.load('Mental_scaled.pkl')

# Load the ordinal encoder 
mental_ord = joblib.load('Mental_ordinal.pkl')

# Load the onehot encoder
mental_ohe = joblib.load('Mental_onehot.pkl')

if sel=="Predict Depression Status":
    col1, col2 = st.columns(2)
    with col1:
        age = st.text_input('**Age**',key='age')
        work_pre = st.slider('**Work Pressure**',int(dep['Work Pressure'].min()),int(dep['Work Pressure'].max()),key='wkp')
        job_sat = st.slider('**Job Satisfaction**',int(dep['Job Satisfaction'].min()),int(dep['Job Satisfaction'].max()),key='jbs')
        work_study_hr = st.slider('**Work/Study Hours**',int(dep['Work/Study Hours'].min()),int(dep['Work/Study Hours'].max()),key='wsh')
        fin_str = st.slider('**Financial Stress**',int(dep['Financial Stress'].min()),int(dep['Financial Stress'].max()),key='fin')
        deg = st.selectbox('**Degree**',DEGREE,index=None)
        suic_tht = st.selectbox('**Have you ever had suicidal thoughts ?**',SUICIDAL_THOUGHTS,index=None)
        
    with col2:
        fam_ill = st.selectbox('**Family History of Mental Illness**',MEDICAL_ILLNESS,index=None)
        gender = st.selectbox('**Gender**',GENDER,index=None)
        city = st.selectbox('**City**',CITY,index=None)
        prof_or_stud = st.selectbox('**Working Professional or Student**',PROF_OR_STUD,index=None)
        profession = st.selectbox('**Profession**',PROFESSION,index=None)
        slp_dur = st.selectbox('**Sleep Duration**',SLEEP,index=None)
        diet = st.selectbox('**Dietary Habits**',DIETARY,index=None)
        
        if st.button("Depression Status"):
            X_unk = np.array([[float(age),work_pre,job_sat,work_study_hr,fin_str,
                               deg,suic_tht,fam_ill,
                                gender,city,prof_or_stud,profession,slp_dur,diet]])
            X_unnum = X_unk[:,0:5]
            X_unord = X_unk[:,5:8]
            X_unohe = X_unk[:,8:]
            X_unnum = mental_sc.transform(X_unnum)
            X_unord = mental_ord.transform(X_unord)
            X_unohe = mental_ohe.transform(X_unohe)
            X_unk_up = np.concatenate([X_unnum,X_unord,X_unohe],axis=1)
            X_unk_tens = torch.tensor(X_unk_up,dtype=torch.float32)

            model_mental.eval()
            with torch.no_grad():
                output = model_mental(X_unk_tens)
                pred = (output >= 0.5).float().squeeze()
                if pred==1:
                    st.write('Yes')
                else:
                    st.write('No')
