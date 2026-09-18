from mcp.server.fastmcp import FastMCP
from query import answer_question


# 1 - Créer une instance de FastMCP

mcp = FastMCP("rag_research")

# 2 - Définir l'outil MCP avec le décorateur @mcp.tool 

@mcp.tool() 
async def search_documents (question: str) -> str :
    """
    Reponds à une question en utilisant les PDFs portant sur différents thèmes indexés. Utilise cet outil quand la question porte sur le contenu de ces documents.
    """
    reponse, sources = answer_question(question)
    sources_texte = "\n".join(sources)
    return f"{reponse}\n\nSources:\n{sources_texte}"


# 3 - Lancement du serveur 

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()    

