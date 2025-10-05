from fastapi import FastAPI
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os 
from dotenv import load_dotenv
import uvicorn 
from youtube_transcript_api import YouTubeTranscriptApi
load_dotenv()
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://vid-vivid-frontend.vercel.app"],  # React app ka URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
class Query(BaseModel):
   statement:str
   id:str

embedding=HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    
    
)
llm= ChatOpenAI(
    model="openai/gpt-4.1-mini" ,
    max_completion_tokens=256,
    api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)

@app.post("/ask")
def view(query:Query):
 
 video_id = query.id
 ytt_api=YouTubeTranscriptApi()
 transcript=[]

 ans = ytt_api.fetch(video_id)
 for i in ans:
   transcript.append(i.text)
   
 


 total_transcript="".join(transcript) #working
 splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
 splitted_text=splitter.split_text(total_transcript) #working
 

 vector_store=Chroma.from_texts(
    texts=splitted_text,
    embedding=embedding,
    collection_name="transcript"
    
)

 retriever=vector_store.as_retriever(search_kwargs={"k":15})


 template= PromptTemplate(
    template="""
    you are a expert adviser who is expert in framing an answer out of given context
    , answer the question asked with reference to the following context only ,
    question - {question}
    context - {text}
    if the context seems insufficient reply saying not sufficient information
    act like you already know everything about the video
    """,
    input_variables=["question", "text"]
)

 input=query.statement
 index=[]
 result=retriever.invoke(input)
 for i in result :
  index.append(i.page_content)
 final_index=" ".join(index) #working
 prompt= template.invoke({"question":input,"text":final_index})
 print(prompt)
 final_ans=llm.invoke(prompt)
 return {
    "answer":final_ans.content
 }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
