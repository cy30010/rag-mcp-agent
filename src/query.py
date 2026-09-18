__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

vecteur2 = Chroma(embedding_function= OllamaEmbeddings(model="nomic-embed-text"), persist_directory="./data/chroma_db")

print(f"Nombre de vecteurs dans la base de données vectorielle : {vecteur2._collection.count()}")


def answer_question(question, model="mistral:7b"):
    retriver = vecteur2.as_retriever(search_kwargs={"k": 4})
    retrived_docs = retriver.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrived_docs)
    prompt = "\n\n".join(["Réponds à la question en utilisant uniquement les informations fournies dans le contexte ci-dessus. Si la réponse n'est pas dans le contexte, réponds par 'Je ne sais pas'.", context, question])
    llm = ChatOllama(model=model)
    reponse = llm.invoke(prompt)
    sources = [f"{doc.metadata.get('source')} - page {doc.metadata.get('page')}" for doc in retrived_docs]
    final_answer = reponse.content, sources
    return final_answer

if __name__ == "__main__":
    result = answer_question("Quels sont les avantages des transformers ?")
    print(result)    