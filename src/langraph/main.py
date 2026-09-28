import re
import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq


# Load environment variables from .env
load_dotenv()


@tool
def search(query: str) -> str:
    """Search the internet for a query. Use this at most once, then answer."""
    print(f"Searching for: {query}")

    # Fake search result for learning/demo purposes
    return "Tokyo weather is sunny."


# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=2,
    timeout=60,
)


# Create LangGraph agent
agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt=(
        "You are a helpful assistant. "
        "Call the search tool at most once, "
        "then answer using the tool result. "
        "Do not search again."
    ),
)


def _retry_seconds(error: Exception, attempt: int) -> int:
    match = re.search(
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        str(error),
        re.I,
    )

    if match:
        return max(1, int(float(match.group(1))) + 1)

    return min(60, 2**attempt)


def invoke_with_retry(prompt: str, attempts: int = 4) -> dict:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=prompt)
                    ]
                },
                {
                    "recursion_limit": 8
                },
            )

        except (ModelAPIError, ModelRateLimitError) as exc:
            last_error = exc

            wait_s = _retry_seconds(exc, attempt)

            print(
                f"Groq unavailable "
                f"(attempt {attempt}/{attempts}). "
                f"Retrying in {wait_s}s..."
            )

            time.sleep(wait_s)

    raise last_error


def main() -> None:
    result = invoke_with_retry(
        "What is the weather in Tokyo?"
    )

    last_message = result["messages"][-1]

    print("\nFinal Answer:")
    print(last_message.content)


if __name__ == "__main__":
    main()

