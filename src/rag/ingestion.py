import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    print("Splitting")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001" ,google_api_key=os.environ.get("GEMINI_API_KEY"))
    
    print("ingestion to pinecone...")
    PineconeVectorStore.from_documents(texts,embeddings,index_name =os.environ.get("INDEX_NAME"))
    print("finish")
