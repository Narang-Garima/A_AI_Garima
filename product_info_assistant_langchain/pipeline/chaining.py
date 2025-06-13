# pipeline/chaining.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from entity.product_schema_model import ProductInfo
from utils.API_loaders import load_api_keys
import logging
def create_chain(model_name: str):
    
    try:
        load_api_keys()
        logging.info("** API Keys loaded successfully from .env")
    except Exception as e:
        logging.error(f"** Failed to load API keys :{e}")


    # Initialize model
    model = ChatGroq(model=model_name, temperature=0.7)

    # Define output parser
    parser = JsonOutputParser(pydantic_object=ProductInfo)

    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a E-Commerce Product Expert. Answer the information related to the products asked for."),
        ("user", "#Format: {format_instructions}\n\n#Question :{question}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # Chain
    chain = prompt | model | parser
    return chain


def get_product_info(chain, product):
    question = f"Give me the name, description, and tentative price (USD) for the product '{product}'."
    raw_response = chain.invoke({"question": question})
    if isinstance(raw_response, dict):
        return ProductInfo(**raw_response)
    return None
