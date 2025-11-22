import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

stdio_server_params = StdioServerParameters(
    command="python",
    args=["/home/leogurdu/LangChain/mcp-crash-course/servers/math_server.py"],
)

async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(
            read_stream=read,
            write_stream=write,
        ) as session:
            await session.initialize()
            print("Initialized stdio client session")
            tools = await load_mcp_tools(session)
            
            agent = create_agent(
                model=llm,
                tools=tools,
            )

            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Cuanto es 4 por 8?")]},
            )
            print("Agent result:", result["messages"][-1].content)
            
if __name__ == "__main__":
    asyncio.run(main())