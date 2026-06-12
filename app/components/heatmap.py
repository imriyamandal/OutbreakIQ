import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

def get_marker_color(risk_level):
    rl = str(risk_level).lower()
    if "critical" in rl:
        return "red"
    elif "high" in rl:
        return "orange"
    elif "medium" in rl:
        return "yellow"
    else:
        return "green"

def render_heatmap(df):
    if df.empty:
        # Fallback empty map
        india_map = folium.Map(
            location=[22.9734, 78.6569],
            zoom_start=5,
            tiles="CartoDB positron"
        )
        st_folium(
            india_map,
            use_container_width=True,
            height=600
        )
        return

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    states_count = len(df["state_ut"].unique())
    zoom = 6 if states_count == 1 else 5

    india_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron"
    )

    marker_cluster = MarkerCluster().add_to(india_map)

    for _, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        state = row.get("state_ut", "Unknown")
        district = row.get("district", "Unknown")
        disease = row.get("disease", "Unknown")
        cases = int(row.get("cases", 0))
        risk_level = row.get("risk_level", "Low")
        prob = float(row.get("outbreak_probability", 0.0))
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; color: #0f172a; line-height: 1.4;">
            <h4 style="margin: 0 0 5px 0; color: #0f766e;">{district}, {state}</h4>
            <b>Disease:</b> {disease}<br>
            <b>Cases:</b> {cases}<br>
            <b>Risk Level:</b> <span style="font-weight: bold; color: {get_marker_color(risk_level)};">{risk_level}</span><br>
            <b>Outbreak Prob:</b> {prob * 100:.1f}%
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            color=get_marker_color(risk_level),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{district} ({disease}): {risk_level} Risk"
        ).add_to(marker_cluster)

    st_folium(
        india_map,
        use_container_width=True,
        height=600
    )