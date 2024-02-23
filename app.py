###### Libraries #######

# Base Libraries
import pandas as pd
import numpy as np

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########

data = pd.read_csv("CarData.csv")

data = data.dropna(axis=0).reset_index(drop=True)

########### Loading Trained Model Files ########
model = joblib.load("cardata_rfreg.pkl")


st.header("Estimation of SellingPrice on Details:")

# Description
st.write("""Built a Predictive model in Machine Learning to estimate Selling price on used cars data.
         Sample Data taken as below shown.
""")

# Data Display
st.dataframe(data.head())
st.write("From the above data , sellingprice is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get the Estimation of sellingprice:")

col1, col2, col3 = st.columns(3) # value inside brace defines the number of splits
col4, col5, col6 = st.columns(3)
col7,col8,col9 = st.columns(3)
col10,col11,col12,col13=st.columns(4)


with col1:
    year = st.number_input("Enter year:")
    st.write(year)

with col2:
    kmdriven = st.number_input("Enter kmdriven:")
    st.write(kmdriven)

with col3: 
    fuel = st.selectbox("Enter fuel:",data.fuel.unique())
    st.write(fuel)

with col4:
    sellertype = st.selectbox("Enter sellertype:",data.seller_type.unique())
    st.write(sellertype)

with col5:
    transmission = st.selectbox("Enter transmission:", data.transmission.unique())
    st.write(transmission)

with col6:
    owner = st.selectbox("Enter owner:", data.owner.unique())
    st.write(owner)

with col7:
    mileage = st.number_input("Enter mileage:")
    st.write(mileage)

with col8:
    engine= st.number_input("Enter engine:")
    st.write(engine)

with col9:
    maxpower= st.number_input("Enter maxpower:")
    st.write(maxpower)

with col10:
    seats = st.number_input("Enter seats:")
    st.write(seats)

data['brand'] = data.name.str.split(expand=True)[0]
    
with col11:
    brand = st.selectbox("Enter select car brand:",data.brand.unique())
    st.write(brand)

with col12:
    torque = st.number_input("Enter torque in nm:")
    st.write(torque)   

with col13:
    speed= st.number_input("Enter speed in rpm:")
    st.write(speed)

###### Predictions #########

if st.button("Estimate"):
    st.write("Data Given:")
    values = [year,kmdriven,fuel,sellertype,transmission,owner,mileage,engine,
    maxpower,seats,brand,torque,speed]
    record =  pd.DataFrame([values],
                           columns = ['year','kmdriven','fuel','sellertype',
                           'transmission','owner','mileage','engine',
                           'maxpower','seats','brand','torquenm','speed'])
    st.dataframe(record)
    price = round(model.predict(record)[0],2)
    price = str(price)+"lakhs"
    st.subheader("Estimated sellingprice:")
    st.subheader(price)







