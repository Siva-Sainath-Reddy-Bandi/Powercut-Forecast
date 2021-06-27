import streamlit as st
from multiapp import MultiApp
from apps import home, data, info, about # import your app modules here

app = MultiApp()

st.markdown("""
# Power outages 
It's important to understand when and why outages occur, as energy is one of the most important resources we have.
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Info", info.app)
app.add_app("Data", data.app)
app.add_app("About", about.app)
# The main app
app.run()


