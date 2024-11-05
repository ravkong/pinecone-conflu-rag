import datetime
import logging
import os
import sys
#import boto3
import datetime, pytz
from pinecone import Pinecone, ServerlessSpec
from atlassian import Confluence
from openai import OpenAI
import langchain
#from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Pinecone
#from langchain.vectorstores.pinecone import PineconeVectorStore
#from langchain_community.llms import OpenAI
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class loadConfluence:
    """
        this class is to load documents from Confluence to pinecone vector DB
    """

    def __init__(self):
        self.pinecone_api_key = '1234'
        self.pinecone_host = "https://empr-fp-56f6ih8.svc.aped-4627-b74a.pinecone.io"
        self.vector_index = 'empr-fp'
        self.namespace = 'financial_planning'
        self.region = 'us-east-1'
        self.embedding_model = 'text-embedding-ada-002'
        self.openai_key='1234'
        self.confluence_api_token = '1234'
        
        os.environ['OPENAI_API_KEY'] = self.openai_key
        os.environ['PINECONE_API_KEY'] = self.pinecone_api_key

        self.client = OpenAI(api_key=self.openai_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key,env=self.region)

    def load_pinecone(self,input_i):


        print("Generating embeddings...")


    # # Generate embeddings using the text-embedding-ada-002 model
        response = self.client.embeddings.create(
            input=input_i,
            model="text-embedding-ada-002"
        )
        #.data[0].embedding

        embeds = [record.embedding for record in response.data]
        print(f'embeds = {embeds}')
    
        # Create an index or connect to an existing one
        index_name = 'openai-embeddings-emp-fp'
        

        print("Loading embeddings to Pinecone...")
        indexes = self.pc.list_indexes()  
        print(f"indexes: {indexes}") 
        index = self.pc.Index(index_name)
        print("Index initialized")

        vectors = []
        for i, e in enumerate(embeds):
            vectors.append({
            "id": f"vec{i}",
            "values": e,
            "metadata": {'text': input_i}
            })

        #vdata = [(f"doc-{i+1}", data['embedding']) for i, data in enumerate(response['data'])]
        #print(f"vdata: {vdata}")
        index.upsert(vectors=vectors, namespace="Financial-Planning")
        print("Embeddings loaded to Pinecone")


    def load_docs(self):
        """
             A condition valuation that gets executes and gets the waiting to refresh count
        :return: the query result
        02/10/2024 : This function os obsolete after account master changes
                """
                #connect to confluence
        # confluence = Confluence(
        #     url='https://conflu-ai.atlassian.net/wiki',  # Confluence Base URL
        #     username='rk.kongara@gmail.com',  # Your email
        #     password=self.confluence_api_token  # API token (recommended) or password
        # )

        space_key = 'Finance'
        confluence_loader = ConfluenceLoader(
        url="https://conflu-ai.atlassian.net/wiki",
        username="rk.kongara@gmail.com",
        api_key=self.confluence_api_token,
        space_key=self.namespace  # List of Confluence page IDs to ingest
        )


        # Check if the connection is successful
        if confluence_loader:
            print("Connection to Confluence successful")
        else:
            print("Connection failed")
        
        documents = confluence_loader.load()

        # 2. Split Documents into Chunks (if needed)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Adjust chunk size based on your model's context length
            chunk_overlap=50
        )

        # Split documents into smaller chunks
        chunks = text_splitter.split_documents(documents)

        #print(chunks)


        # 3. Generate Embeddings for Each Chunk
        # Initialize the embedding model (you can use OpenAI, Hugging Face, etc.)
        embeddings = OpenAIEmbeddings()

        # Create an index if it doesn't exist
        index_name = "confluence-docs"


        index = self.pc.Index(index_name)

        print("delete records from index")

        try:
            index.delete(delete_all=True, namespace=self.namespace)
            print("Records deleted successfully")
        except Exception as e:
            print("Error in deleting records")
            print(e)

        vectorstore_from_docs = PineconeVectorStore.from_documents(
        chunks,index_name=index_name,embedding=embeddings,namespace=self.namespace)
        print("Documents successfully ingested into Pinecone!")




    def load_docs_v2(self):
        """
             A condition valuation that gets executes and gets the waiting to refresh count
        :return: the query result
        02/10/2024 : This function os obsolete after account master changes
                """
                #connect to confluence
        # confluence = Confluence(
        #     url='https://conflu-ai.atlassian.net/wiki',  # Confluence Base URL
        #     username='rk.kongara@gmail.com',  # Your email
        #     password=self.confluence_api_token  # API token (recommended) or password
        # )

        space_key = 'Finance'
        confluence_loader = ConfluenceLoader(
        url="https://conflu-ai.atlassian.net/wiki",
        username="rk.kongara@gmail.com",
        api_key=self.confluence_api_token,
        space_key=space_key  # List of Confluence page IDs to ingest
        )

        # Check if the connection is successful
        if confluence_loader:
            print("Connection to Confluence successful")
        else:
            print("Connection failed")
        
        documents = confluence_loader.load()

        # 2. Split Documents into Chunks (if needed)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Adjust chunk size based on your model's context length
            chunk_overlap=50
        )

        # Split documents into smaller chunks
        chunks = text_splitter.split_documents(documents)

        #print(chunks)

        docs_key = []
        texts= []
        i=1

        for doc in chunks:
            texts.append(doc.page_content)
            docs_key.append('doc-' + str(i))
            i=i+1
            if i>2:
                break


        
        #print(texts)
        #print(docs_key)

        # Generate embeddings
        #response = self.client.embeddings.create(input=texts, model="text-embedding-ada-002")
        embeddings = OpenAIEmbeddings()
        embedded_docs = embeddings.embed_documents([doc.page_content for doc in chunks])

        vectors_to_upsert = []
        val=1
        for i, doc in enumerate(texts):
            vectors_to_upsert.append({
            'id': f'doc_{i}',  # Use a unique identifier for each document
            'values': embedded_docs[i],
            'metadata': {'doc-'+str(val)}
        })
            val=val+1
            
        index = self.pc.Index("confluence-docs")
 
        # Upsert the vectors
        index.upsert(vectors=vectors_to_upsert)


        #       #print(type(response))
        # #embedding_vectors = [data['embedding'] for data in response['data']]
        # embedding_vectors = [record.embedding for record in response['data']]
        # #embedding_vectors = [data['embedding'] for data in response['data']]
        # embedding_model = OpenAIEmbeddings(openai_api_key=self.openai_key)
        # vector_store = Pinecone(index_name = "confluence-docs", embedding_function=embedding_model.embed_query)
        # vectors_to_upsert = list(zip(docs_key, embedding_vectors))
        # vector_store.upsert(vectors=vectors_to_upsert)
        # print("Documents successfully ingested into Pinecone!")







    def query_pine_cone(self,query_text):

        #openai.api_key = self.openai_key

        
        # Generate an embedding for a query text
        #query_text = "what is the fee structure?"
        response = self.client.embeddings.create(
            input=query_text,
            model="text-embedding-ada-002"
        ).data[0].embedding
        query_embedding = response

        print("Query embedding generated")
        print(query_embedding)

        index_name = 'openai-embeddings-emp-fp'
        pc = Pinecone(api_key=self.pinecone_api_key,env=self.region)

        print("Loading embeddings to Pinecone...")

        index = pc.Index(index_name)


        # Query the Pinecone index with the query embedding
        query_results = index.query(
            queries=[query_embedding],
            top_k=5  # Return top 5 most similar results
        )
        print(query_results)



def main():
    obj = loadConfluence()
    #pc = Pinecone(api_key=pcx.pinecone_api_key,env=pcx.region)
    #print(pc.list_indexes())
    #exit(0)
    print("preparing data is started")
    #text="For Empower to manage your investments you should deposit at least $50,000 into empower managed account. The fee structure is as follows: less than $100,000 = 0.02% of total account value, between $100,000 and $250,000 = 0.01% of total account value, greater than $250,000 = 0.005% of total account value"

    #embed= pcx.prepare_data()
    print("preparing data is completed")
    obj.load_docs() 
    #pcx.query_pine_cone("what is the fee structure?")
    exit(0)

if __name__ == '__main__':
    main()
