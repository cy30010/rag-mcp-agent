import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client 

async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["src/mcp_server.py"],
    )


    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.call_tool(
            name="search_documents",
            arguments={"question": "Quels sont les avantages des transformers ?"}
            )

            print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())

