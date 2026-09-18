__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import shutil
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

#1 - Extraction de texte des fichiers PDF 
loader = PyPDFDirectoryLoader("./data/pdfs")
docs = loader.load()

print(len(docs))
print ( docs[37].page_content)
print (docs[37].metadata)


#2 - Découpage du texte en morceaux plus petits (Chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 150)

chunks = splitter.split_documents(docs)
print(len(chunks))
 

#3 - Embeddings et stockage dans ChromaDB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists("./data/chroma_db"):
    shutil.rmtree("./data/chroma_db") #base réinitialisée à chaque exécution du script pour éviter les doublons
    
vecteur = Chroma.from_documents(chunks, embeddings, persist_directory="./data/chroma_db")

print(f"Nombre de vecteurs dans la base de données vectorielle : {vecteur._collection.count()}")