import streamlit as st

def app():
    st.title('Content')

    st.write('The dataset contains outage data from year 2000 up until 2014, with the following information:')

    st.text('- Event Description: Reason of the outage (e.g. Vandalistm, Severe Weather, etc)')
    st.text('- Year: Year of the outage (e.g. 2014)')
    st.text('- Month: Month of the outage (e.g. November)')
    st.text('- Date Event Began: Date of the outage (e.g. 11/13/2017)')
    st.text('- Time Event Began: Time the outage was registered (e.g. 15:05:00)')
    st.text('- Date of Restoration: Date the outage was resolved')
    st.text('- Time of Restoration: Time the outage was resolved')
    st.text('- Respondent: The company that acted upon the outage')
    st.text('- Geografic Area: Region of the outage (e.g. Missisippi, Texas, New York, Salt Lake City, etc)')
    st.text('- NERG Region: (NERC refers to the North American Electricity Reliability Corporation, formed to ensure the reliability of the grid)')
    st.text('- Demand Loss (MW): How much energy was not transmited/consumed during the outage')
    st.text('- Number of customers affected: How many consumers (e.g. homes, offices, industry, etc) were left to their devices')
    st.text('- Tags: Summary event description (e.g. wild fire, vandalism, severe weather)')



