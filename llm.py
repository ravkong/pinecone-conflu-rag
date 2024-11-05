# Description: This script is used to retrieve text from OpenAI API using the prompt provided by the user.
    
import streamlit as st
from openai import OpenAI
#import pinecone
import os
import sys
import json
#from ollama import Ollama
from pinecone import Pinecone, ServerlessSpec

pinecone_api_key = '1234'
pinecone_host = "https://empr-fp-56f6ih8.svc.aped-4627-b74a.pinecone.io"
vector_index = 'empr-fp'
namespace = 'financial_planning'
region = 'us-east-1'
embedding_model = 'ada-002'
openai_key='1234'
confluence_api_token = '1234'

def embed(input_text):
    """
    Generate embeddings for the given input text.
    """
    client = OpenAI(api_key=openai_key)
    
    response = client.embeddings.create(
    input=input_text,
    model="text-embedding-ada-002"  # Specify the correct model name for embeddings
    )
    
    response = response.data[0].embedding
    print(f"Generated embedding: {response}")
    return response

def query_pinecone(query_vector,query_text,namespace):
    """
    Query Pinecone to get relevant documents for the given query text.
    """
    # Generate embeddings for the query (using a model of your choice, or if available via AWS Bedrock)
    # For simplicity, let's assume the embeddings are already generated for this example.
    # Example: query_embedding = generate_embeddings(query_text)

    # This should be the embedding you generated from a model like OpenAI, Hugging Face, or AWS Bedrock.
    #query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]  # Replace with actual embeddings

    pc = Pinecone(api_key=pinecone_api_key, env=region)

    #index_name = "openai-embeddings-emp-fp"  # Replace with your Pinecone index name
    index_name = 'confluence-docs'
    #namespace = 'financial_planning'
    index = pc.Index(index_name)

    #query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Query Pinecone index with the generated embeddings
    search_results = index.query(
        namespace=namespace,
        vector=query_vector,
        top_k=5,  # Retrieve top 5 relevant documents
        include_metadata=True
    )

    #context = "\n".join([result['metadata']['text'] for result in search_results['matches']])
    #context = "\n".join([result['text'] for result in search_results['matches']])
    contexts = [item['metadata']['text'] for item in search_results['matches']]
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query_text
    print("Retrieved Context:", augmented_query)
    return augmented_query
    #return search_results

def generate_response_ollama(context):
    """
    Generate a response to the given prompt.
    """
    # Use the prompt to generate a response using a model of your choice (e.g., OpenAI, Hugging Face, etc.)
    # For simplicity, let's assume the response is already generated for this example.
    # Example: response = generate_response(prompt)

    client = Ollama(host='http://localhost:11434')  # Update host and port if different

    # Define your prompt by including the context
    prompt = f"Given the following context:\n{context}\n\nWhat is your response to this query?"

    # Generate response using Ollama
    response = client.generate(model="llama", prompt=prompt)

    # Print the generated response
    print("Generated Response:", response['choices'][0]['text'])

    return response

def generate_response_gpt(context):
    """
    Generate a response to the given prompt.
    """

    # system message to 'prime' the model
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know".
    """
    # Define your prompt by including the context
    #prompt = f"Given the following context:\n{context}\n\nWhat is your response to this query?"
    #prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Generate response using OpenAI GPT-3
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=prompt,
    #     temperature=0.5,
    #     max_tokens=100
    # )
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
    model="gpt-4o",  # You can use "gpt-4" or any other available model
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": context}
    ]
    )

    # Print the generated response
    #response = response['choices'][0]['message']['content']
    print("Generated Response:", response)
    response = response.choices[0].message.content
    print("Generated Response:", response)
    #response = response.choices[0].text.strip()
    return response   