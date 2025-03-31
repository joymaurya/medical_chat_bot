from langchain_groq import ChatGroq
from src.prompt import system_prompt
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
import socket
import uvicorn
from dotenv import load_dotenv
import os
from fastapi import FastAPI

app=FastAPI()

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




@app.get("/")
def start():
    return f"Hey, I am Running !! {socket.gethostname()} "

@app.post("/medical_bot")
def chat_bot(input:str):
    chain_input={"input":input}
    response=final_chain.invoke(chain_input)
    return response["answer"]


if __name__=="__main__":
    uvicorn.run(app="app:app",host="0.0.0.0",port=5000,reload=True)