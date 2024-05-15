import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st
import plotly.express as px
import seaborn as sns
from scipy.stats import zscore
from PIL import Image

# Sample earthquake dataset (replace with your own dataset)
df = pd.read_csv('C:/Users/User/Documents/Sample Dashboard/Sample 1/earthquake_1995-2023.csv')

# Preprocessing
df.drop(['alert', 'continent'], axis=1, inplace=True)

# Split 'location' to extract 'country'
df[['city', 'country']] = df['location'].str.split(',', n=1, expand=True)

# Handle missing 'country' values
df['country'] = df['country'].fillna(df['city'])

# Parse date_time with correct format
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M', dayfirst=True)

# Extract year, month, day from date_time
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day

# Reorder columns
df = df[['date_time', 'day', 'month', 'year'] + [col for col in df.columns if col not in ['date_time', 'day', 'month', 'year']]]

# Remove outliers
threshold = 2.5
z_scores = zscore(df['sig'])
df_no_outliers = df[(z_scores < threshold) & (z_scores > -threshold)]

threshold = 3
z_scores_n = zscore(df_no_outliers['dmin'])
df_n = df_no_outliers[(z_scores_n < threshold) & (z_scores_n > -threshold)]

z_scores_n = zscore(df_n['gap'])
df_n1 = df_n[(z_scores_n < threshold) & (z_scores_n > -threshold)]

z_scores_n = zscore(df_n1['depth'])
df = df_n1[(z_scores_n < threshold) & (z_scores_n > -threshold)]

# Streamlit app functions
def intro():
    st.header("Earthquake Visualization and Prediction")
    st.subheader("By: Lim Vern Sin (0133235), Sarah Darlyna Bt Mohd Radzi (0134768)")
    st.markdown("Earthquakes have been the leading causes of death by natural disasters. "
                "Recent magnitude-7.2 earthquake event dated 17 November 2023 hit Southern Philippines causing devastating effects on property and lives.")
    
    video_file = open('C:/Users/User/Documents/Sample Dashboard/Sample 1/Strong 6.7 earthquake hits Philippines without any tsunami threat - Friday, November 17, 2023.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.caption('Source: https://www.youtube.com/watch?v=zJ-ucEG3xAE&ab_channel=euronews')

    st.markdown("A strong earthquake is measured based on its magnitude. The stronger the magnitude, the deadlier the earthquake scale. The Richter scale is a logarithmic scale that measures the magnitude of an earthquake based on the energy released.")
    image2 = Image.open('C:/Users/User/Documents/Sample Dashboard/Sample 1/richter scale.png')
    st.image(image2, caption='Source: https://www.britannica.com/science/Richter-scale')

    st.markdown("90% of the world's earthquakes and 80% of the world's deadliest earthquakes are located in the Ring Of Fire (S. Gonzaga, 2023). The Ring of Fire is a string of volcanoes and sites of seismic activity, or earthquakes, around the edges of the Pacific Ocean.")
    image = Image.open('C:/Users/User/Documents/Sample Dashboard/Sample 1/pacific-ring-of-fire.jpg')
    st.image(image, caption='Source: https://education.nationalgeographic.org/resource/plate-tectonics-ring-fire/')

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
        title={
            'text': ' ',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'family': 'Arial', 'color': 'black'}
        }
    )
    fig.update_traces(marker=dict(size=10))
    st.subheader("Earthquake Occurrences Worldwide Based On Magnitude Since 1995")
    st.plotly_chart(fig, theme="streamlit")
    st.markdown("The world map and the scatter traces on it are the earthquake events that happen in that specific location. The darker it is, the higher the magnitude of the earthquake event")

def top_20():
    count = df['country'].value_counts().head(20)
    st.subheader("Top 20 Countries With Most Frequent Earthquake Occurrences And Their Magnitude")
    sns.set(style="darkgrid")
    sbd = sns.countplot(x='country', color='#66FFFF', hue='magnitude', data=df, order=count.index, dodge=False)
    sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.markdown("This visualization narrows down the scope of which countries have the highest number of earthquake events, along with the magnitude of these events.")

def tsunami():
    st.subheader('Number Of Tsunami Occurrences Based On Countries')
    selected_option = st.selectbox('Select Event', ['Tsunami', 'No Tsunami'])

    if selected_option == 'Tsunami':
        filtered_data_1 = df[df['tsunami'] == 1]
        st.subheader('Countplot For Tsunami Occurrences Based On Countries')
        sns.set_theme(style="darkgrid")
        sns.set(font_scale=0.5)
        sbd = sns.countplot(x='country', data=filtered_data_1, color='#0066FF')
        sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
        st.pyplot()
        st.markdown("This visualization shows which countries experienced tsunamis after an earthquake event. The top 3 countries that frequently experience tsunamis are often island nations (Indonesia, Papua New Guinea, Vanuatu).")

    elif selected_option == 'No Tsunami':
        filtered_data = df[df['tsunami'] == 0]
        st.subheader('Countplot For No Tsunami Occurrences Based On Countries')
        sns.set_theme(style="darkgrid")
        sns.set(font_scale=0.4)
        sbd = sns.countplot(x='country', data=filtered_data, color='#0066FF')
        sbd.set_xticklabels(sbd.get_xticklabels(), rotation=90)
        st.pyplot()
        st.markdown("This visualization shows which countries did not experience tsunamis after an earthquake event. More countries that are not island nations, such as Mexico, fall into this category.")

def magnitude_count():
    map_graph()
    top_20()
    fig_n = px.line(df, 'date_time', 'magnitude', labels={'index': 'Year', 'date_time': 'Year'})
    fig_n.update_traces(marker=dict(line=dict(color='#FFFFFF', width=2)))
    fig_n.update_traces(textposition='top center')
    fig_n.update_layout(title_text=' ', title_x=0.5)
    st.subheader("Magnitude Of Earthquake Events From 1995-2023")
    st.plotly_chart(fig_n, theme="streamlit")
    st.markdown("The time series visualization above shows the trend of earthquake magnitudes from 1995 until 2023. The strongest earthquake recorded during this period was in 2004 with a magnitude of 9.0.")
    tsunami()

def predict_magnitude():
    X = df[['latitude', 'longitude', 'depth']]
    y = df['magnitude']
    model = Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor())])
    model.fit(X, y)

    st.subheader('Earthquake Magnitude Prediction For The Following Year')
    st.markdown("The prediction model was trained with a Random Forest Regression model with 73% accuracy.")
    
    lat = st.number_input('Enter Latitude:', min_value=-90, max_value=90, value=0)
    long = st.number_input('Enter Longitude:', min_value=-180, max_value=180, value=0)
    
    if st.button('Predict Magnitude'):
        new_location = [[lat, long, 0]]
        predicted_magnitude = model.predict(new_location)[0]
        st.write(f'Predicted Magnitude: {predicted_magnitude:.2f}')

page_names_to_funcs = {
    "Introduction": intro,
    "Descriptive": magnitude_count,
    "Prediction": predict_magnitude
}

demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
