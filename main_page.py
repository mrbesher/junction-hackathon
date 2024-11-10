import streamlit as st

st.set_page_config(
    page_title="CodeVelhot Junction 2024",
    page_icon="📜",
    layout="wide",
)

st.title("Zero Friction powered by CodeVelhot.")
st.sidebar.success('Solutions to empower digital democracy.')

with st.sidebar:
    st.page_link('main_page.py', label='Home', icon='🏠')
    st.page_link('pages/Explore_Data.py', label='Explore Data', icon='🔍')
    st.page_link('pages/Multimodal_Polis.py', label='Chat', icon='👋')

st.page_link("pages/Multimodal_Polis.py", label='Start!')
