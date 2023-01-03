import streamlit as st

from default.read_defaults import *

tab1, tab2 = st.tabs(['Usual', 'About data'])
with tab1:
    st.write(texts['medicalAnalysisMainText'])
with tab2:
    st.write(texts['contextData'])
    st.write(texts['aboutData'])
