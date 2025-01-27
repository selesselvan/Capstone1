import streamlit as st
import requests
import pickle


# Load country mapping
with open('country_mapping.pkl', 'rb') as f:
    country_mapping = pickle.load(f)

# Reverse mapping for display
countries = list(country_mapping.keys())

# Load City mapping
with open('city_mapping.pkl', 'rb') as f:
    city_mapping = pickle.load(f)

# Reverse mapping for display
cities = list(city_mapping.keys())


# Load co_aqi_category mapping
with open('co_aqi_category_mapping.pkl', 'rb') as f:
    co_aqi_category_mapping = pickle.load(f)

# Reverse mapping for display
co_aqi = list(co_aqi_category_mapping.keys())

# Load no2_aqi_category mapping
with open('no2_aqi_category_mapping.pkl', 'rb') as f:
    no2_aqi_category_mapping = pickle.load(f)

# Reverse mapping for display
no2_aqi = list(no2_aqi_category_mapping.keys())


# Load ozone_aqi_category mapping
with open('ozone_aqi_category_mapping.pkl', 'rb') as f:
    ozone_aqi_category_mapping = pickle.load(f)

# Reverse mapping for display
ozone_aqi = list(ozone_aqi_category_mapping.keys())

# Load pm25_aqi_category mapping
with open('pm25_aqi_category_mapping.pkl', 'rb') as f:
    pm25_aqi_category_mapping = pickle.load(f)

# Reverse mapping for display
pm25_aqi = list(pm25_aqi_category_mapping.keys())



# Page Configuration
st.set_page_config(
    page_title="AQI Prediction Dashboard",  
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #4a4a4a;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title('Air Quality Index (AQI) Prediction Dashboard')

# Informative Expander
with st.expander("What is Air Quality Index (AQI)?"):
    st.markdown("""
    
    The AQI is a critical tool used to communicate how polluted the air currently is or how polluted it is forecast to become. 
    
    #### Key Features:
    - **Measure of Air Pollution**: Provides a standardized indicator of air quality
    - **Key Pollutants Tracked**:
        - Particulate Matter (PM2.5)
        - Ozone (O₃)
        - Carbon Monoxide (CO)
        - Nitrogen Dioxide (NO₂)
    
    #### AQI Categories:
    - **0-50**: Good 
    - **51-99**: Moderate 
    - **100-149**: Unhealthy for Sensitive Groups 
    - **150-200**: Unhealthy 
    - **201-297**: Very Unhealthy 
    - **300-500**: Hazardous
    """)


# Country Selection
selected_country = st.selectbox('Select Country:', countries)
selected_country_code = country_mapping[selected_country]

#st.write(f"Selected Country: {selected_country}")
#st.write(f"Country Code: {selected_country_code}")

# City Selection
selected_city = st.selectbox('City:', cities)
selected_city_code = city_mapping[selected_city]

# AQI value
aqi_value = st.number_input('AQI Value', min_value=0, max_value=500, value=0)

# co_aqi_category Selection
selected_co_aqi = st.selectbox('CO Category:', co_aqi)
selected_co_aqi_code = co_aqi_category_mapping[selected_co_aqi]

# co_aqi_value

co_aqi_value = st.number_input('CO Value', min_value=0, max_value=500, value=0)

# no2_aqi_category Selection
selected_no2_aqi = st.selectbox('NO2 Category:', no2_aqi)
selected_no2_aqi_code = no2_aqi_category_mapping[selected_no2_aqi]

# no2_aqi_value

no2_aqi_value = st.number_input('NO2 Value', min_value=0, max_value=500, value=0)

# ozone category Selection
selected_ozone_aqi = st.selectbox('Ozone Category:', ozone_aqi)
selected_ozone_aqi_code = ozone_aqi_category_mapping[selected_ozone_aqi]

# ozone_aqi_value
ozone_aqi_value = st.number_input('Ozone Value', min_value=0, max_value=500, value=0)

# pm25_aqi_value
pm25_aqi_value = st.number_input('PM2.5 Value', min_value=0, max_value=500, value=0)

#pm25_aqi_category
selected_pm25_aqi = st.selectbox('PM25 Category:', pm25_aqi)
selected_pm25_aqi_code = pm25_aqi_category_mapping[selected_pm25_aqi]

if st.button('Predict Air Quality Index Category'):
    input_data = {
        'Country': int(selected_country_code),
        'City': int(selected_city_code),
        'AQI_Value': float(aqi_value),
        'CO_AQI_Value': float(co_aqi_value),
        'CO_AQI_Category': int(selected_co_aqi_code),
        'Ozone_AQI_Value': float(ozone_aqi_value),
        'Ozone_AQI_Category': int(selected_ozone_aqi_code),
        'NO2_AQI_Value': float(no2_aqi_value),
        'NO2_AQI_Category': int(selected_no2_aqi_code),
        'PM2.5_AQI_Value': float(pm25_aqi_value),
        'PM2.5_AQI_Category': int(selected_pm25_aqi_code)
    }
    
    #st.write("Input Data:", input_data)
    
    try:
        response = requests.post('http://localhost:9696/predict', json=input_data)
        response.raise_for_status()
        
        result = response.json()
        st.success(f"Predicted AQI Category: {result['prediction']}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        st.error(f"Response Content: {e.response.text}")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")


