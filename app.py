# app.py
import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import datetime
from folium.plugins import Draw




# -------------------------
# Initialization
# -------------------------
st.set_page_config(page_title="ReGenVision - Land Health", layout="wide")
st.title("🌍 ReGenVision — Land Health Explorer")

# Initialize Earth Engine (safe)
def init_ee():
    try:
        ee.Initialize(project='siol-degradation')
        return True, None
    except Exception as e:
        try:
            ee.Initialize()  # fallback to default legacy project
            return True, None
        except Exception as e2:
            return False, str(e2)


ee_ok, ee_err = init_ee()
if not ee_ok:
    st.error("❌ Earth Engine init failed: " + str(ee_err))
    st.info("Run `earthengine authenticate` locally and ensure your account has permission or remove project arg in ee.Initialize().")
    st.stop()

# Optional: load saved model for land-health scoring (if present)
MODEL_PATH = "model_xgb.pkl"
@st.cache_resource
def load_model(path=MODEL_PATH):
    try:
        m = joblib.load(path)
        return m
    except Exception:
        return None

model = load_model()

# -------------------------
# Helper: create folium map with draw tools
# -------------------------
DEFAULT_CENTER = [-3.8, 38.4]  # Taita Taveta center
DEFAULT_BBOX = [38.2, -4.3, 38.6, -3.2]  # lon_min, lat_min, lon_max, lat_max

def make_map():
    fmap = folium.Map(location=DEFAULT_CENTER, zoom_start=9)
    draw = Draw(export=True, filename='aoi.geojson', draw_options={'polyline': False, 'circlemarker': False})
    draw.add_to(fmap)
    return fmap

# -------------------------
# UI: Controls
# -------------------------
st.sidebar.header("📍 Select area & period")
use_default = st.sidebar.checkbox("Use default Taita Taveta bbox", value=True)

start_date = st.sidebar.date_input("Start date", value=datetime.date(2023,1,1))
end_date = st.sidebar.date_input("End date", value=datetime.date(2024,12,31))
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")

# Map area selection
st.sidebar.markdown("Draw a polygon on the map (use Draw tools) or use default bbox.")
fmap = make_map()
map_out = st_folium(fmap, width=700, height=450, returned_objects=["last_active_drawing", "all_drawings", "last_clicked"])

# Extract AOI geometry
def parse_aoi(map_out, use_default):
    if use_default:
        lon_min, lat_min, lon_max, lat_max = DEFAULT_BBOX
        geom = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        return geom, "Default bbox (Taita Taveta)"
    # check last_active_drawing then all_drawings
    drawing = map_out.get("last_active_drawing") or (map_out.get("all_drawings") and list(map_out.get("all_drawings").values())[-1])
    if drawing:
        geojson = drawing
        # folium draw geojson structure -> convert to ee.Geometry
        coords = geojson.get("geometry", {}).get("coordinates", None)
        if coords is None:
            return None, None
        # depending on polygon vs multi
        try:
            # use the first polygon ring
            poly = ee.Geometry.Polygon(coords[0] if isinstance(coords[0][0], (list,)) else coords)
        except Exception:
            # attempt direct import
            poly = ee.Geometry(geojson)
        return poly, "User-drawn polygon"
    # if clicked point exists
    last_clicked = map_out.get("last_clicked")
    if last_clicked:
        lat = last_clicked["lat"]
        lon = last_clicked["lng"]
        point_geom = ee.Geometry.Point([lon, lat]).buffer(500)  # 500 m buffer
        return point_geom, f"Point buffer around ({lat:.4f}, {lon:.4f})"
    return None, None

aoi_geom, aoi_label = parse_aoi(map_out, use_default)

if aoi_geom is None:
    st.warning("No AOI selected. Draw a polygon or enable default bbox.")
    st.stop()

st.sidebar.success(f"AOI: {aoi_label}")

# -------------------------
# Earth Engine helpers
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_monthly_ndvi_and_rain(_aoi, start, end, scale=30):
    aoi = _aoi


    # convert to ee Dates
    start_date = ee.Date(str(start))
    end_date = ee.Date(str(end))

    # Sentinel-2 NDVI time series (use SR if available)
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR')
          .filterBounds(aoi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
         )

    def add_ndvi(img):
        nd = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands(nd).copyProperties(img, ['system:time_start'])

    s2_ndvi = s2.map(add_ndvi).select('NDVI')

    # CHIRPS daily precipitation -> we will aggregate monthly
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(aoi).filterDate(start_date, end_date)

    # Build list of month ranges
    def months_list(start_dt, end_dt):
        s = pd.to_datetime(start_dt)
        e = pd.to_datetime(end_dt)
        rng = pd.date_range(s, e, freq='MS')
        months = []
        for dt in rng:
            ms = dt.strftime('%Y-%m-%d')
            me = (dt + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            months.append((ms, me))
        return months

    months = months_list(start, end)

    rows = []
    for ms, me in months:
        ms_date = ee.Date(ms)
        me_date = ee.Date(me).advance(1, 'day')  # include end

        # NDVI monthly composite (median)
        ndvi_month = s2_ndvi.filterDate(ms_date, me_date).median()
        ndvi_reduced = ndvi_month.reduceRegion(ee.Reducer.mean(), aoi, scale=scale)
        ndvi_val = ndvi_reduced.getInfo().get('NDVI', None)

        # Rainfall monthly sum
        rain_month = chirps.filterDate(ms_date, me_date).sum()
        rain_reduced = rain_month.reduceRegion(ee.Reducer.sum(), aoi, scale=5000)  # CHIRPS coarse
        rain_val = rain_reduced.getInfo().get('precipitation', None)

        rows.append({
            'month_start': ms,
            'month_end': me,
            'ndvi': ndvi_val,
            'rainfall': rain_val
        })

    df = pd.DataFrame(rows)
    # convert month_start to datetime
    df['month'] = pd.to_datetime(df['month_start'])
    # drop NaNs gracefully
    df['ndvi'] = pd.to_numeric(df['ndvi'], errors='coerce')
    df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def fetch_static_layers(_aoi):
    aoi = _aoi

    """Fetch static contextual layers: mean SOC (SoilGrids), mean slope (SRTM)"""
    # SoilGrids SOC mean (project path may differ; adjust if necessary)
    try:
        soc_img = ee.Image('projects/soilgrids-isric/soc_mean')  # or 'ISRIC/SoilGrids'
        soc_mean = soc_img.reduceRegion(ee.Reducer.mean(), aoi, scale=250).getInfo()
        soc_val = soc_mean.get('soc_mean', None)
    except Exception:
        soc_val = None

    # SRTM slope
    try:
        elev = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(elev)
        slope_mean = slope.reduceRegion(ee.Reducer.mean(), aoi, scale=250).getInfo()
        slope_val = slope_mean.get('slope', None)
    except Exception:
        slope_val = None

    return {'soc': soc_val, 'slope': slope_val}

# -------------------------
# Fetch data button
# -------------------------
if st.button("🔎 Analyze selected area"):
    with st.spinner("Fetching time series from Earth Engine (this may take a while for large AOIs)..."):
        df_ts = fetch_monthly_ndvi_and_rain(aoi_geom, str(start_date), str(end_date))
        ctx = fetch_static_layers(aoi_geom)

    st.success("✅ Data retrieved")

    # Basic metrics
    mean_ndvi = float(df_ts['ndvi'].mean()) if not df_ts['ndvi'].isna().all() else None
    ndvi_trend = None
    if df_ts['ndvi'].notna().sum() >= 3:
        # simple linear trend (slope)
        x = np.arange(len(df_ts))
        mask = ~df_ts['ndvi'].isna()
        if mask.sum() >= 2:
            coef = np.polyfit(x[mask], df_ts['ndvi'][mask], 1)
            ndvi_trend = float(coef[0])
    rainfall_total = float(df_ts['rainfall'].sum()) if not df_ts['rainfall'].isna().all() else None

    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean NDVI", f"{mean_ndvi:.3f}" if mean_ndvi is not None else "N/A")
    col2.metric("NDVI trend (slope per month)", f"{ndvi_trend:.5f}" if ndvi_trend is not None else "N/A")
    col3.metric("Total Rainfall (period)", f"{rainfall_total:.1f} mm" if rainfall_total is not None else "N/A")

    # show static layers
    st.write("### Contextual static layers")
    st.write(f"- Soil Organic Carbon (SOC) mean: {ctx.get('soc')}")
    st.write(f"- Slope (degrees) mean: {ctx.get('slope')}")

    # Charts: NDVI and Rainfall
    st.write("### Time series")
    fig_ndvi = px.line(df_ts, x='month', y='ndvi', markers=True, title="NDVI over time")
    fig_ndvi.update_layout(yaxis_title="NDVI", xaxis_title="Month")

    fig_rain = px.bar(df_ts, x='month', y='rainfall', title="Monthly Rainfall (mm)")
    fig_rain.update_layout(yaxis_title="Rainfall (mm)", xaxis_title="Month")

    st.plotly_chart(fig_ndvi, use_container_width=True)
    st.plotly_chart(fig_rain, use_container_width=True)

    # -------------------------
    # Optional: Model prediction (if model exists)
    # -------------------------
    if model is not None:
        st.subheader("🤖 Land Health Score Prediction")

        # simple example: use last NDVI, mean rainfall, static layers
    if mean_ndvi is not None and rainfall_total is not None:
        features = pd.DataFrame([{
            'Rainfall': rainfall_total or 0,
            'SOC': ctx.get('soc') or 0,
            'Slope': ctx.get('slope') or 0
        }])
        try:
            score = float(model.predict(features)[0])
            st.success(f"🌱 Predicted Land Health Score: **{score:.2f} / 1.00**")
        except Exception as e:
            st.warning(f"⚠️ Model prediction failed: {e}")
    else:
        st.info("Not enough data for model prediction.")
    st.info("No model file (`model_xgb.pkl`) found — skipping prediction step.")

    # -------------------------
    # Data download
    # -------------------------
    st.write("### 📥 Download data")
    csv = df_ts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download time series as CSV",
        data=csv,
        file_name="regenvision_landhealth_timeseries.csv",
        mime="text/csv"
    )

    st.write("✅ Analysis complete.")