import streamlit as st

pg = st.navigation([st.Page("Translator.py"), st.Page('source_code.py')])
pg.run()