#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV


# In[2]:


st.title('Electric Motor Temperature: Motor Speed Prediction')
st.write('Random Forest Regressor')


# In[3]:


st.sidebar.header('User Input Parameters')


# In[4]:


def user_input_features():
    Ambient = st.sidebar.number_input("Insert ambient")
    Coolant = st.sidebar.number_input("Insert Coolant value")
    u_d = st.sidebar.number_input("Insert u_d")
    u_q = st.sidebar.number_input("Insert u_q")
    Torque = st.sidebar.number_input("Insert Torque")
    i_d = st.sidebar.number_input("Insert i_d")
    i_q = st.sidebar.number_input("Insert i_q")
    pm = st.sidebar.number_input("Insert pm")
    stator_yoke = st.sidebar.number_input("Insert stator_yoke")
    stator_tooth = st.sidebar.number_input("Insert stator_tooth")
    stator_winding = st.sidebar.number_input("Insert stator_winding")
    data = {'ambient':Ambient,
            'coolant':Coolant,
            'u_d':u_d,
            'u_q':u_q,
            'torque':Torque,
            'i_d':i_d,
            'i_q':i_q,
            'pm':pm,
            'stator_yoke':stator_yoke,
            'stator_tooth':stator_tooth,
            'stator_winding':stator_winding}
    features = pd.DataFrame(data,index = [0])
    return features 


# In[5]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[6]:


motor = pd.read_csv("temperature_data.csv")
motor.drop(["profile_id"],inplace=True,axis = 1)
motor = motor.dropna()


# In[7]:


X = motor.drop(['motor_speed'],axis=1)
Y = motor.iloc[:,4]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
clf = RandomForestRegressor()
clf.fit(X_train,Y_train) 


# In[8]:


Prediction = clf.predict(X_train)
Motor_speed = clf.predict(X_train) 


# In[9]:


st.subheader('Motor speed Prediction')
st.write(Motor_speed)


# In[ ]:




