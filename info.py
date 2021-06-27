import streamlit as st

def app():
    st.title('Inspiration')

    st.write("When looking at this dataset, one can raise several questions:")

    st.text('- What causes outages?')
    st.text('- What are the most common causes?')
    st.text('- Where and when is it more common?')
    st.text('- When is it more impactful?')

    st.write('With that, perhaps, the biggest question of all can be answered: what could be done to improve the situation?')

    st.title('Acknowledgements')

    st.write("This data was compiled by the reporter [Jordan Wirfs-Brock](http://insideenergy.org/author/jbrock/) and the original post can be found at [Inside Energy](http://insideenergy.org/2014/08/18/data-explore-15-years-of-power-outages/#comment-3862651149).The source material is publicly available at [ENERGY](https://www.oe.netl.doe.gov/OE417_annual_summary.aspx)")
