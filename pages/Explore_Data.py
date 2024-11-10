import streamlit as st
import plotly.express as px
import pandas as pd

from helpers import read_json

st.title("Insights from Polis")

# json_data = read_json(r'data/topic_visualization.json')
# df = pd.DataFrame(json_data['points'])
# df = df[~(df['topic']==-1)]

# fig = px.scatter(df, x='x', y='y', color='topic_label', hover_data=['document'])

# st.plotly_chart(fig)
with st.sidebar:
    st.page_link("main_page.py", label="Home", icon="🏠")
    st.page_link("pages/Explore_Data.py", label="Explore Data", icon="🔍")
    st.page_link("pages/Multimodal_Polis.py", label="Chat", icon="👋")

st.markdown(
    """
    <iframe src="http://127.0.0.1:8050/" width="900" height="700"></iframe>
    """,
    unsafe_allow_html=True,
)
