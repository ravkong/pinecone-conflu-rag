import datetime
import logging
import os
import sys
#import boto3
import datetime, pytz
from pinecone import Pinecone, ServerlessSpec
from atlassian import Confluence
from openai import OpenAI

logger = logging.getLogger(__name__)


class loadPc:
    """
        this class is to pinecone vector DB
    """

    def __init__(self):
        self.pinecone_api_key = '1234'
        self.pinecone_host = "https://empr-fp-56f6ih8.svc.aped-4627-b74a.pinecone.io"
        self.vector_index = 'empr-fp'
        self.namespace = 'financial_planning'
        self.region = 'us-east-1'
        self.embedding_model = 'ada-002'
        self.openai_key='1234'
        self.confluence_api_token = '1234'
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


    def prepare_data(self):
        """
             A condition valuation that gets executes and gets the waiting to refresh count
        :return: the query result
        02/10/2024 : This function os obsolete after account master changes
                """
                #connect to confluence
        confluence = Confluence(
            url='https://conflu-ai.atlassian.net/wiki',  # Confluence Base URL
            username='rk.kongara@gmail.com',  # Your email
            password=self.confluence_api_token  # API token (recommended) or password
        )

        # Check if the connection is successful
        if confluence:
            print("Connection to Confluence successful")
        else:
            print("Connection failed")
        
        space_key = 'Finance'
        pages = confluence.get_all_pages_from_space(space=space_key, start=0, limit=10)
        for page in pages:
            print(f"Page title: {page['title']}")
            print(f"Page ID: {page['id']}")
            print(f"Page URL: {page['_links']['webui']}")
        
        page_content = confluence.get_page_by_id(589846, expand='body.view')
        #print(page_content['title'])
        #print(page_content['body']['view']['value'])

        # Set your OpenAI API key
    

        print("Generating embeddings...")


        # # Generate embeddings using the text-embedding-ada-002 model
        response = self.client.embeddings.create(
             input=page_content['body']['view']['value'],
             model="text-embedding-ada-002"
         ).data[0].embedding
        
        return response

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
    pcx = loadPc()
    #pc = Pinecone(api_key=pcx.pinecone_api_key,env=pcx.region)
    #print(pc.list_indexes())
    #exit(0)
    print("preparing data is started")
    text="For Empower to manage your investments you should deposit at least $100,000 into empower managed account. The fee structure is as follows: less than $100,000 = 0.02% of total account value, between $100,000 and $250,000 = 0.01% of total account value, greater than $250,000 = 0.005% of total account value"

    #embed= pcx.prepare_data()
    print("preparing data is completed")
    pcx.load_pinecone(text) 
    #pcx.query_pine_cone("what is the fee structure?")
    exit(0)






    # Create a serverless index
    index_name = pcx.vector_index
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 

        # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        print("Index not ready yet, waiting...")
        time.sleep(1)
        
    index = pc.Index(index_name)

    print(index)
    exit(0)


    index.upsert(
        vectors=[
            {"id": "vec1", "values": [1.0, 1.5]},
            {"id": "vec2", "values": [2.0, 1.0]},
            {"id": "vec3", "values": [0.1, 3.0]},
        ],
        namespace="ns-conflu-1"
    )

    index.upsert(
        vectors=[
            {"id": "vec1", "values": [1.0, -2.5]},
            {"id": "vec2", "values": [3.0, -2.0]},
            {"id": "vec3", "values": [0.5, -1.5]},
        ],
        namespace="ns-conflu-2"
    )

    #print("describing the indexes")

    #print(index.describe_index_stats())
    print("listing the indexes")
    print(pc.list_indexes())

    query_results1 = index.query(
    namespace="ns-conflu-1",
    vector=[1.0, 1.5],
    top_k=3,
    include_values=True
)

    query_results2 = index.query(
        namespace="ns-conflu-2",
        vector=[1.0,-2.5],
        top_k=3,
        include_values=True
    )


    print(query_results1)
    print(query_results2)

    pc.delete_index('example-index')
    print("listing the indexes")
    print(pc.list_indexes())

""" 
    # Create Index
    index_name = "text-embedding-ada-002"

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    index = pc.Index(index_name)

    # Insert vectors
    vectors = [
        (f"doc_{i}", [0.1] * 1536)
        for i in range(100)
    ] """


if __name__ == '__main__':
    main()
