import streamlit as st
from reccomender import recommend_assessments

st.title("SHL Assessment Recommendation System")

# Input form
query = st.text_area("Enter job description, query, or URL:")
max_duration = st.number_input("Max Duration (minutes, optional)", min_value=0, value=0, step=5)
if max_duration == 0:
    max_duration = None

if st.button("Recommend Assessments"):
    if query:
        recommendations = recommend_assessments(query, max_duration)
        if recommendations:
            st.write("### Recommended Assessments")
            st.table(recommendations)
        else:
            st.error("No assessments found matching your criteria.")
    else:
        st.error("Please enter a query or job description.")