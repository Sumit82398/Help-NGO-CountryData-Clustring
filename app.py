import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Lets load joblib instances over here 
with open ('Pipeline.joblib','rb')as file:
    Preprocess = joblib.load(file)

with open ('model.joblib','rb')as file:
    model = joblib.load(file)

# Lets take the inputs from the user 
st.title('Help NGO Organization')
st.subheader('This Application will help to identify the devlopment.category of a country using Social economic factors.Original data has been clustered using k-mean')

# Lets take the inputs
gdpp = st.number_input('Enter the GPPP of a country (GDP per population)')
income = st.number_input('Enter income per population')
imports = st.number_input('Imports of goods and services per capita. Given as age of the GDP per capita')
exports = st.number_input('Exports of goods and services per capita. Given as age of the GDP per capita')
inflation = st.number_input('Inflation: The measurement of the annual growth rate of the Total GDP')
life_expcy = st.number_input('Life_Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain the same')
fert = st.number_input('Fertility: The number of children that would be born to each woman if the current age-fertility rates remain the same.')
health = st.number_input('Health: Total health spending per capita. Given as age of GDP per capita')
child_mort = st.number_input('Child_mort: Death of children under 5 years of age per 1000 live births')


# Lets create a input List
input_list = [child_mort,exports,health,imports,income,inflation,life_expcy,fert,gdpp]

# Lets create a final input List 
final_input_list = Preprocess.transform([input_list])

# Lets predict output
if st.button('Predict'):
    prediction = model.predict(final_input_list)[0]
    if prediction == 0:
        st.success('Developing')
    elif prediction == 1:
        st.success('Developed')
    else:
        st.error('Underdeveloped')