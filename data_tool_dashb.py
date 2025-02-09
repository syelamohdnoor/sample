import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from PIL import Image

# Load dataset
df = pd.read_csv('earthquake_1995-2023.csv')

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
    
    st.markdown("""
        Earthquakes have been the leading causes of death by natural disasters.  
        A recent magnitude-7.2 earthquake on **November 17, 2023**, hit Southern Philippines, causing devastating effects on property and lives.
    """)

    video_file = open('Strong 6.7 earthquake hits Philippines without any tsunami threat - Friday, November 17, 2023.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.caption('Source: https://www.youtube.com/watch?v=zJ-ucEG3xAE&ab_channel=euronews')

    st.markdown("""
        A strong earthquake is measured based on its magnitude. The stronger the magnitude, the deadlier the earthquake scale.  
        The **Richter scale** is a logarithmic scale that measures the magnitude of an earthquake based on the energy released.
    """)
    image2 = Image.open('richter scale.png')
    st.image(image2, caption='Source: https://www.britannica.com/science/Richter-scale')

    st.markdown("""
        **90% of the world's earthquakes and 80% of the world's deadliest earthquakes** occur in the **Ring of Fire**.  
        The **Ring of Fire** is a string of volcanoes and sites of seismic activity around the edges of the Pacific Ocean.
    """)
    image = Image.open('pacific-ring-of-fire.jpg')
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
    st.markdown("The darker the color, the higher the earthquake magnitude.")

def top_20():
    count = df['country'].value_counts().head(20)
    st.subheader("Top 20 Countries With Most Frequent Earthquake Occurrences And Their Magnitude")
    
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 5))  # Explicit figure creation
    sbd = sns.countplot(x='country', color='#66FFFF', hue='magnitude', data=df, order=count.index, dodge=False, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    st.pyplot(fig)  # Pass figure explicitly
    st.markdown("This visualization highlights the countries with the highest number of earthquake occurrences.")

def tsunami():
    st.subheader('Number Of Tsunami Occurrences Based On Countries')
    selected_option = st.selectbox('Select Event', ['Tsunami', 'No Tsunami'])

    if selected_option == 'Tsunami':
        filtered_data_1 = df[df['tsunami'] == 1]
        st.subheader('Countplot For Tsunami Occurrences Based On Countries')

        sns.set_theme(style="darkgrid")
        sns.set(font_scale=0.5)

        fig, ax = plt.subplots(figsize=(10, 5))  # Explicit figure creation
        sns.countplot(x='country', data=filtered_data_1, color='#0066FF', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        st.pyplot(fig)  # Pass figure explicitly
        st.markdown("This visualization shows the frequency of tsunamis occurring after earthquakes in different countries.")

# Streamlit Navigation
page_names_to_funcs = {
    "Introduction": intro,
    "Global Earthquake Map": map_graph,
    "Top 20 Countries with Earthquakes": top_20,
    "Tsunami Occurrences": tsunami,
}

st.sidebar.title("Navigation")
demo_name = st.sidebar.radio("Go to", list(page_names_to_funcs.keys()))

# Run selected page
page_names_to_funcs[demo_name]()
