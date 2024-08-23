from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import folium
import os

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('NewAncientTemples.csv')
df['Description'] = df['Description'].fillna('')
df['Coordinates'] = df['Coordinates'].fillna('(0, 0)')

# Combine Description for content-based filtering
df['content'] = df['Description']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

# Function to parse coordinates
def parse_coordinates(coord_str):
    try:
        coords = eval(coord_str)
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            return [float(coords[0]), float(coords[1])]
    except:
        pass
    return [np.nan, np.nan]

# Define recommendation function
def get_recommendations(description=None, name=None):
    recommendations = pd.DataFrame()

    if description or name:
        desc_sim = cosine_similarity(tfidf.transform([description if description else ""]), tfidf_matrix)
        name_sim = cosine_similarity(tfidf.transform([name if name else ""]), tfidf_matrix)
        
        # Combine similarity scores
        combined_scores = (desc_sim[0] + name_sim[0]) / 2

        # Filter out recommendations with similarity <= 0.0
        high_sim_indices = np.where(combined_scores > 0.0)[0]

        if len(high_sim_indices) > 0:
            high_sim_scores = combined_scores[high_sim_indices]
            desc_scores = sorted(list(zip(high_sim_indices, high_sim_scores)), key=lambda x: x[1], reverse=True)
            desc_scores = desc_scores[:10]  # Top 10 recommendations
            desc_indices = [i[0] for i in desc_scores]
            recommendations = df.iloc[desc_indices][['templeName', 'Coordinates', 'Description']]

    recommendations = recommendations.drop_duplicates(subset='templeName')
    return recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    description = request.form.get('description')
    recommendations = get_recommendations(description=description)
    return render_template('recommend.html', recommendations=recommendations)

@app.route('/show_map')
def show_map():
    # Create folium map for all locations
    coordinates = pd.DataFrame(df['Coordinates'].apply(parse_coordinates).tolist(), columns=['Latitude', 'Longitude'])
    df[['Latitude', 'Longitude']] = coordinates
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row['templeName'],
            icon=folium.Icon(icon='info-sign')
        ).add_to(m)

    map_html = m._repr_html_()

    return render_template('map.html', map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
