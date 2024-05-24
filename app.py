import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle


def load_data():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    df.dropna(subset='bmi', inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    return df

df = load_data()

def load_model():
    with open('pickle_model.pkl', 'rb') as file:
        data = pickle.load(file)

    return data

data = load_model()

clf = data['model']
encoder = data['encoder']


st.set_page_config(page_title='Stoke prediction', layout='centered')

st.title('Stroke Prediction App')
st.subheader('Information to predict whether an individual is likely to have stroke or not')

st.subheader('User Inputs')

gender = st.selectbox('Gender', df.gender.unique().tolist())
age = st.number_input('Age', min_value=18)
hypertension = st.selectbox('Hypertensive? : Select 1 for YES, 0 for NO', [0,1])
heart = st.selectbox('Heart disease? : Select 1 for YES, 0 for NO', [0,1])
married = st.selectbox('Ever married?',df.ever_married.unique().tolist())
work = st.selectbox('Work type', df.work_type.unique().tolist())
residence = st.selectbox('Residence', df.Residence_type.unique().tolist())
glucose = st.number_input('Average Fasting Glucose level (mg/dL)', placeholder='Input average glucose level...')
bmi = st.number_input('BMI', placeholder='Input BMI...')
smoke = st.selectbox('Please select smoking status', df.smoking_status.unique().tolist())

result = st.button('Predict')



new_input = pd.DataFrame({'gender': [gender],
                          'age': [age],
                          'hypertension':[hypertension],
                          'heart_disease':[heart],
                          'ever_married':[married],
                          'work_type': [work],
                          'Residence_type': [residence],
                          'avg_glucose_level':[glucose],
                          'bmi': [bmi],
                          'smoking_status': [smoke]
})


features = df.select_dtypes(include='object').columns.to_list()



if result:
    trans_input = encoder.transform(new_input[features].values)
    new_input = pd.concat([new_input,
                    pd.DataFrame(trans_input, columns=encoder.get_feature_names_out(features))], axis=1).drop(features, axis=1)
    
    clf.predict(new_input)
    
    if clf.predict(new_input)[0] == 1:
        st.write('There is risk of stroke')
    else:
        st.write('No risk of stroke')


