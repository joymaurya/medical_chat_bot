from langchain_groq import ChatGroq
from src.prompt import system_prompt
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from dotenv import load_dotenv
import os
from fastapi import FastAPI

load_dotenv()

os.environ["GROP_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"]=os.getenv("PINECONE_API_KEY")

embedding=download_hugging_face_embeddings()

docsearch=PineconeVectorStore.from_existing_index(
    index_name="med-bot",
    embedding=embedding
)

retriver=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

model_name="llama-3.3-70b-versatile"

model = ChatGroq(temperature=1, model_name="llama-3.3-70b-versatile")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("user","{input}")
    ]
)

qna_chain=create_stuff_documents_chain(llm=model,prompt=prompt)
final_chain=create_retrieval_chain(retriver,qna_chain)
input={"input":"What is Acne?"}

response=final_chain.invoke(input)

print(response["answer"])