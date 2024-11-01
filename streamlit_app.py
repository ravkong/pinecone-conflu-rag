# Description: This script is used to retrieve text from OpenAI API using the prompt provided by the user.
    
import streamlit as st
from ingest_docs import loadConfluence
from llm import embed, query_pinecone, generate_response_gpt

# Sidebar Navigation
st.sidebar.title("Navigation")

section = st.sidebar.radio("Choose a section:", ("HR", "Finance","Onboarding", "Infrastructure", "AppReviews","Support","Architecture"))

if section == "HR":
    st.write("HR Knowledge Base")
elif section == "Finance":
    st.write("Financial Advise Knowledge Base")    
elif section == "Onboarding":
    st.write("Onboarding knowledge Base")
elif section == "AppReviews":
    st.write("App Reviews")
elif section == "Infrastructure":
    st.write("Infrastructure Knowledge Base")
elif section == "Support":
    st.write("Support Knowledge Base")
else:
    st.write("Architecture Knowledge Base")        

if st.sidebar.button('Update Vector Database', key="update_vector_database"):
    with st.spinner('Updating the vector database...'):   
        load_pinecone = loadConfluence()
        load_pinecone.namespace=section
        load_pinecone.load_docs()
        # Add your logic to update the vector database here
        st.success("Vector database updated successfully!")  

st.title('The Empower AI')
st.subheader('The Real(time) AI Assistant for Everyone!')
prompt = st.text_input('Enter your question:', key="question_input")

#col1, col2 = st.columns([1, 1])

if st.button('Generate response', key="generate_response"):
    with st.spinner('Generating response...'):
        vector = embed(prompt)
        augmented_query = query_pinecone(vector,prompt,section)
        response = generate_response_gpt(augmented_query)
        st.text_area('Generated response:', value=response, height=200)  


                