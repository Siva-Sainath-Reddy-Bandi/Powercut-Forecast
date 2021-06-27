import streamlit as st

def app():
    st.subheader('Data: Explore 15 Years Of Power Outages')

    st.write('This database details 15 years of power outages across the United States, compiled and standardized from annual data available at from the Department of Energy.')

    st.subheader('What this data tells us?')

    st.write('The data tells us what government agencies have reported to the Department of Energy about grid outages in their region or sector.  It also tells us information about those outages – when and where they occurred, how long they lasted, what caused them, what actions were taken, how many people lost power.')

    st.subheader('Limitations of the data')

    st.write('While this database tells us a lot about grid vulnerabilities, it does have limitations. Only major electricity providers and operators are required to report outages, so this database is not comprehensive.')

    st.write('The data comes in all kinds of non-standardized forms and information isn’t reported consistently or comprehensively. For example, if the description of an event simple says “fire,”  we don’t know if that means a wild fire or an electrical fire. Some events are missing information all together: They have no data for the number of customers who lost power, or the location is vague or difficult to decipher.')

