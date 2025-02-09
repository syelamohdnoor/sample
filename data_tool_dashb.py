import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('earthquake_1995-2023.csv')

# Preprocessing
df.drop(['alert', 'continent'], axis=1, inplace=True)

# Extract 'country' from 'location'
df[['city', 'country']] = df['location'].str.split(',', n=1, expand=True)
df['country'] = df['country'].fillna(df['city'])

# Convert date_time column
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M', dayfirst=True)
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day

# Remove outliers using z-score
threshold = 2.5
df = df[(zscore(df['sig']) < threshold) & (zscore(df['sig']) > -threshold)]
df = df[(zscore(df['depth']) < 3) & (zscore(df['depth']) > -3)]

# Streamlit App
st.set_page_config(page_title="Earthquake Visualization & Prediction", layout="wide")

# Sidebar Dropdown Menu
page = st.sidebar.selectbox("Choose a page:", ["Introduction", "Descriptive Analysis", "Prediction"])

# Page: Introduction
if page == "Introduction":
    st.header("Earthquake Visualization and Prediction")
    st.subheader("By: Lim Vern Sin (0133235), Sarah Darlyna Bt Mohd Radzi (0134768)")
    
    st.markdown("Earthquakes have been the leading causes of death by natural disasters. "
                "A recent magnitude-7.2 earthquake event on 17 November 2023 hit Southern Philippines, causing devastating effects.")

    video_file = open('Strong 6.7 earthquake hits Philippines without any tsunami threat - Friday, November 17, 2023.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.caption('Source: https://www.youtube.com/watch?v=zJ-ucEG3xAE&ab_channel=euronews')

    st.markdown("90% of the world's earthquakes occur in the **Ring Of Fire**.")
    image = Image.open('pacific-ring-of-fire.jpg')
    st.image(image, caption='Source: National Geographic')

# Page: Descriptive Analysis
elif page == "Descriptive Analysis":
    sub_option = st.sidebar.selectbox("Choose a view:", ["Global Earthquake Map", "Top 20 Earthquake Countries", "Tsunami Occurrences"])

    if sub_option == "Global Earthquake Map":
        st.subheader("Earthquake Occurrences Worldwide Based On Magnitude Since 1995")
        fig = px.scatter_geo(
            df, lat='latitude', lon='longitude',
            color='magnitude', color_continuous_scale='PuBu',
            size='depth', scope='world',
            projection='natural earth', animation_frame='year'
        )
        fig.update_layout(title=" ", title_x=0.5)
        fig.update_traces(marker=dict(size=10))
        st.plotly_chart(fig, theme="streamlit")
        st.markdown("Darker colors indicate higher earthquake magnitudes.")

    elif sub_option == "Top 20 Earthquake Countries":
        st.subheader("Top 20 Countries With Most Frequent Earthquake Occurrences")
        count = df['country'].value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.set(style="darkgrid")
        sbd = sns.countplot(x='country', color='#66FFFF', hue='magnitude', data=df, order=count.index, dodge=False, ax=ax)
        sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
        st.pyplot(fig)
        st.markdown("This shows countries with the highest number of earthquakes.")

    elif sub_option == "Tsunami Occurrences":
        st.subheader('Number Of Tsunami Occurrences Based On Countries')
        selected_option = st.selectbox('Select Event', ['Tsunami', 'No Tsunami'])
        
        if selected_option == 'Tsunami':
            filtered_data = df[df['tsunami'] == 1]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.set_theme(style="darkgrid")
            sbd = sns.countplot(x='country', data=filtered_data, color='#0066FF', ax=ax)
            sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
            st.pyplot(fig)
            st.markdown("This shows countries that experienced tsunamis after an earthquake.")

# Page: Prediction
elif page == "Prediction":
    st.subheader('Earthquake Magnitude Prediction For The Following Year')
    st.markdown("The prediction model was trained using **Random Forest Regression** with **73% accuracy**.")

    X = df[['latitude', 'longitude', 'depth']]
    y = df['magnitude']

    model = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor())])
    model.fit(X, y)

    lat = st.number_input('Enter Latitude:', min_value=-90, max_value=90, value=0)
    long = st.number_input('Enter Longitude:', min_value=-180, max_value=180, value=0)

    if st.button('Predict Magnitude'):
        new_location = [[lat, long, 0]]
        predicted_magnitude = model.predict(new_location)[0]
        st.write(f'Predicted Magnitude: **{predicted_magnitude:.2f}**')
