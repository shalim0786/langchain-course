import os
import warnings
from dotenv import load_dotenv

# Forcefully .env load karein
load_dotenv(override=True)

warnings.filterwarnings("ignore")

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def main() -> None:
    print("Executing LLM call and sending traces...")

    information = """
    Elon Reeve Musk (born June 28, 1971) is a businessman, industrialist and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX.
    """

    summary_template = """
    Given the following information about a person:
    {information}

    Please provide:
    1. Short summary
    2. Two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)   
    
    chain = summary_prompt_template | llm
    
    result = chain.invoke({"information": information})
    
    print("\n--- Output ---")
    
    # Simple IF-ELSE condition:
    if isinstance(result.content, list):
        print(result.content[0]["text"])
    else:
        print(result.content)
        
    print("--------------")

if __name__ == "__main__":
    main()