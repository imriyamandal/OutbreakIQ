import folium
import pandas as pd

df = pd.read_csv(
    "ml/data/final/ml_data.csv"
)

m = folium.Map(
    location=[22.5,78.9],
    zoom_start=5
)

for _, row in df.iterrows():

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=8,
        popup=f"{row['district']}",
        fill=True
    ).add_to(m)

m.save(
    "dashboard/heatmap.html"
)