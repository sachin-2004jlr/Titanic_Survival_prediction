import streamlit as st
import pandas as pd
import joblib
import json
import requests
from streamlit_lottie import st_lottie
from confluent_kafka import Consumer
import plotly.express as px
import plotly.graph_objects as go
import uuid

st.set_page_config(page_title="Titanic Pro Systems", layout="wide", page_icon="⚓")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except:
    pass

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_radar = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_w51pcehl.json")

try:
    with open('assets/config.json') as f:
        config = json.load(f)
except:
    config = {"app_name": "Titanic AI", "version": "3.0", "theme": "Default"}

@st.cache_resource
def load_model():
    return joblib.load('models/titanic_model.pkl')

try:
    model = load_model()
except:
    st.error("❌ Model missing! Run 'python train_model.py' first.")
    st.stop()

with st.sidebar:
    if lottie_radar:
        st_lottie(lottie_radar, height=150, key="radar_anim")
    st.title(config['app_name'])
    st.caption(f"v{config['version']} | {config['theme']}")
    st.divider()
    page = st.radio("NAVIGATION", ["📊 Analytics Dashboard", "🧠 AI Prediction", "⚡ Live Kafka Stream"])

def extract_features_from_stream(data):
    title = 'Mr'
    if 'Mrs.' in data.get('Name',''): title = 'Mrs'
    elif 'Miss.' in data.get('Name',''): title = 'Miss'
    sibsp, parch = int(data.get('SibSp', 0)), int(data.get('Parch', 0))
    return pd.DataFrame({
        'Pclass': [int(data.get('Pclass', 3))],
        'Sex': [data.get('Sex', 'male')],
        'Age': [float(data.get('Age', 28))],
        'Fare': [float(data.get('Fare', 10.0))],
        'Embarked': [data.get('Embarked', 'S')],
        'Title': [title],
        'FamilySize': [sibsp + parch + 1],
        'IsAlone': [1 if (sibsp + parch + 1) == 1 else 0],
        'Deck': ['M']
    })

if page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Historical Data Analysis")
    try:
        df = pd.read_csv('data/train.csv')
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("TOTAL PASSENGERS", len(df))
        col2.metric("SURVIVAL RATE", f"{df['Survived'].mean()*100:.1f}%")
        col3.metric("AVG FARE", f"${df['Fare'].mean():.2f}")
        col4.metric("AVG AGE", f"{df['Age'].mean():.1f}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Survival by Class")
            fig = px.bar(df.groupby('Pclass')['Survived'].mean().reset_index(), x='Pclass', y='Survived', color='Pclass')
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Age Distribution")
            fig2 = px.histogram(df, x="Age", color="Survived", nbins=30)
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
    except:
        st.error("Data file not found in data/ folder.")

elif page == "🧠 AI Prediction":
    st.markdown("## 🧠 Neural Survival Predictor")
    with st.expander("⚙️ PASSENGER PARAMETERS", expanded=True):
        c1, c2 = st.columns(2)
        pclass = c1.selectbox("Class", [1, 2, 3])
        sex = c1.selectbox("Gender", ["male", "female"])
        age = c2.slider("Age", 0, 100, 25)
        fare = c2.number_input("Fare", 0, 500, 50)

    if st.button("RUN PREDICTION"):
        input_data = pd.DataFrame({'Pclass': [pclass], 'Sex': [sex], 'Age': [age], 'Fare': [fare], 'Embarked': ['S'], 'Title': ['Mr'], 'FamilySize': [1], 'IsAlone': [1], 'Deck': ['M']})
        prob = float(model.predict_proba(input_data)[0][1])
        
        c1, c2 = st.columns([1,2])
        with c1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text': "Survival Chance"}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#06b6d4"}}))
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("### Result Analysis")
            if prob > 0.5: st.success(f"## LIKELY TO SURVIVE ({prob*100:.1f}%)")
            else: st.error(f"## UNLIKELY TO SURVIVE ({prob*100:.1f}%)")

elif page == "⚡ Live Kafka Stream":
    st.markdown("## ⚡ Real-Time Data Pipeline")
    if st.button("START LISTENING"):
        conf = {'bootstrap.servers': 'localhost:9092', 'group.id': f"st-{uuid.uuid4()}", 'auto.offset.reset': 'latest'}
        consumer = Consumer(conf)
        consumer.subscribe(['titanic_stream'])
        
        st.info("🟢 CONNECTION ACTIVE")
        placeholder = st.empty()
        data_log = []
        
        try:
            while True:
                msg = consumer.poll(0.5)
                if msg is None or msg.error(): continue
                data = json.loads(msg.value().decode('utf-8'))
                prob = float(model.predict_proba(extract_features_from_stream(data))[0][1])
                data['Status'] = "Alive" if prob > 0.5 else "Deceased"
                data['Confidence'] = f"{prob*100:.1f}%"
                data_log.insert(0, data)
                
                with placeholder.container():
                    k1, k2, k3 = st.columns(3)
                    k1.metric("PASSENGER", data['Name'])
                    k2.metric("PREDICTION", data['Status'])
                    k3.metric("CONFIDENCE", data['Confidence'])
                    st.dataframe(pd.DataFrame(data_log).head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            consumer.close()