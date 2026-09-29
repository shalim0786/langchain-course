from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily = TavilyClient()
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.

    Args:
        query: The query to search for

    Returns:
        The search result
    """
    print(f"Searching for {query}")

    return tavily.search(query=query)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

tools = [search]

agent = create_agent(
    model=llm,
    tools=tools
)


def main():
    print("Hello from LangChain course")

    result = agent.invoke(
        {
            "messages": [   
                HumanMessage(
                    content="What is the weather in Tokyo?"
                )
            ]
        }
    )

    print(result)


if __name__ == "__main__":
    main()