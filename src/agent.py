import asyncio
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["src/mcp_server.py"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            llm = ChatOllama(model="qwen3.6:latest")
            llm_with_tools = llm.bind_tools(tools)
            question = "Quels sont les avantages des transformers ?"
            debut=time.time()
            response = llm_with_tools.invoke(question)
            duree=time.time()-debut
            print(f"Temps de décision du LLM : {duree:.2f} secondes")

            print(response)
            print(response.tool_calls)

            if response.tool_calls:
                debut=time.time()
                tool_call = response.tool_calls[0]
                tool_result = await session.call_tool(
                    name=tool_call["name"],
                    arguments=tool_call["args"]
                )
                duree=time.time()-debut
                print(f"Temps d'appel de l'outil : {duree:.2f} secondes")
                print("Résultat de l'outil :")
                print(tool_result.content[0].text)


                messages = [HumanMessage(content=question)]
                messages.append(response)
                messages.append(ToolMessage(content=tool_result.content[0].text, tool_call_id=tool_call["id"]))

                debut=time.time()
                final_response = llm_with_tools.invoke(messages)
                duree=time.time()-debut
                print(f"Temps de synthèse finale : {duree:.2f} secondes")
                print(final_response.content)
            else:
                print("Le LLM a répondu directement, sans utiliser d'outil :")
                print(response.content)

if __name__ == "__main__":
    asyncio.run(main())