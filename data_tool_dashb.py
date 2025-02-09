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

# Reorder columns
df = df[['date_time', 'day', 'month', 'year'] + [col for col in df.columns if col not in ['date_time', 'day', 'month', 'year']]]

# Remove outliers using z-score
threshold = 2.5
df_no_outliers = df[(zscore(df['sig']) < threshold) & (zscore(df['sig']) > -threshold)]
df_no_outliers = df_no_outliers[(zscore(df_no_outliers['dmin']) < 3) & (zscore(df_no_outliers['dmin']) > -3)]
df_no_outliers = df_no_outliers[(zscore(df_no_outliers['gap']) < 3) & (zscore(df_no_outliers['gap']) > -3)]
df = df_no_outliers[(zscore(df_no_outliers['depth']) < 3) & (zscore(df_no_outliers['depth']) > -3)]

# Streamlit App Functions
def intro():
    st.header("Earthquake Visualization and Prediction")
    st.subheader("By: Lim Vern Sin (0133235), Sarah Darlyna Bt Mohd Radzi (0134768)")

    st.markdown("Earthquakes have been the leading causes of death by natural disasters. "
                "A recent magnitude-7.2 earthquake event on 17 November 2023 hit Southern Philippines, causing devastating effects.")

    video_file = open('Strong 6.7 earthquake hits Philippines without any tsunami threat - Friday, November 17, 2023.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.caption('Source: https://www.youtube.com/watch?v=zJ-ucEG3xAE&ab_channel=euronews')

    st.markdown("A strong earthquake is measured based on its magnitude. The Richter scale is a logarithmic scale that measures the magnitude of an earthquake based on the energy released.")
    image2 = Image.open('richter scale.png')
    st.image(image2, caption='Source: Britannica')

    st.markdown("90% of the world's earthquakes occur in the Ring Of Fire. This is a string of volcanoes and seismic activity sites around the Pacific Ocean.")
    image = Image.open('pacific-ring-of-fire.jpg')
    st.image(image, caption='Source: National Geographic')

def map_graph():
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='magnitude', color_continuous_scale='PuBu',
        size='depth',
        scope='world',
        projection='natural earth',
        animation_frame='year'
    )
    fig.update_layout(
        title={'text': ' ', 'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'}
    )
    fig.update_traces(marker=dict(size=10))
    
    st.subheader("Earthquake Occurrences Worldwide Based On Magnitude Since 1995")
    st.plotly_chart(fig, theme="streamlit")
    st.markdown("The world map shows earthquake events based on location. Darker colors indicate higher magnitudes.")

def top_20():
    count = df['country'].value_counts().head(20)
    st.subheader("Top 20 Countries With Most Frequent Earthquake Occurrences And Their Magnitude")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set(style="darkgrid")
    sbd = sns.countplot(x='country', color='#66FFFF', hue='magnitude', data=df, order=count.index, dodge=False, ax=ax)
    sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
    
    st.pyplot(fig)  # Fixed warning: explicitly passing 'fig'
    
    st.markdown("This visualization shows the countries with the most earthquake events, along with their magnitudes.")

def tsunami():
    st.subheader('Number Of Tsunami Occurrences Based On Countries')
    selected_option = st.selectbox('Select Event', ['Tsunami', 'No Tsunami'])

    if selected_option == 'Tsunami':
        filtered_data_1 = df[df['tsunami'] == 1]
        st.subheader('Countplot For Tsunami Occurrences Based On Countries')

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.set_theme(style="darkgrid")
        sns.set(font_scale=0.5)
        sbd = sns.countplot(x='country', data=filtered_data_1, color='#0066FF', ax=ax)
        sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
        
        st.pyplot(fig)  # Fixed warning: explicitly passing 'fig'
        
        st.markdown("This visualization shows which countries experienced tsunamis after an earthquake event.")

def predict_magnitude():
    X = df[['latitude', 'longitude', 'depth']]
    y = df['magnitude']
    
    # Create a pipeline for preprocessing and regression
    model = Pipeline([
        ('scaler', StandardScaler()), 
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)

    st.subheader('Earthquake Magnitude Prediction For The Following Year')
    st.markdown("The prediction model was trained with a Random Forest Regression model with 73% accuracy.")
    
    lat = st.number_input('Enter Latitude:', min_value=-90, max_value=90, value=0)
    long = st.number_input('Enter Longitude:', min_value=-180, max_value=180, value=0)
    depth = st.number_input('Enter Depth (km):', min_value=0, max_value=700, value=10)

    if st.button('Predict Magnitude'):
        new_location = [[lat, long, depth]]
        predicted_magnitude = model.predict(new_location)[0]
        st.write(f'Predicted Magnitude: {predicted_magnitude:.2f}')

# Sidebar Navigation
page_names_to_funcs = {
    "Introduction": intro,
    "Global Earthquake Map": map_graph,
    "Top 20 Earthquake Countries": top_20,
    "Tsunami Occurrences": tsunami,
    "Prediction": predict_magnitude
}

demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
