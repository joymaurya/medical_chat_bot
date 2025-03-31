from src.helper import download_hugging_face_embeddings,load_pdf_file,text_split
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")


extracted_data=load_pdf_file(path="data/")

splitted_data=text_split(extracted_data)

embedding=download_hugging_face_embeddings()


def create_pinecode_db(index_name):
    pc = Pinecone()
    pc.create_index(
        name=index_name,
        dimension=384, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )

index_name="med-bot"

create_pinecode_db(index_name)

docsearch=PineconeVectorStore.from_documents(
    documents=splitted_data,
    index_name=index_name,  
    embedding=embedding
)