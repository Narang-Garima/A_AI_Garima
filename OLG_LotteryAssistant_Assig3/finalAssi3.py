# Imports
from pydantic import BaseModel,Field
from typing import TypedDict, Annotated, Sequence
from langchain.schema import BaseMessage
import operator, os, requests
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, SeleniumURLLoader
from pinecone import Pinecone
from pinecone import ServerlessSpec
from uuid import uuid4
from langchain_pinecone import PineconeVectorStore
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from langchain_core.documents import Document
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
import streamlit as st



# Check and set environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY= os.getenv("TAVILY_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
pinecone_api= os.getenv("PINECONE_API_KEY")

# Langsmith Tracking and tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


# LLM model
model = ChatOpenAI(
    model="gpt-3.5-turbo")


# Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "User messages"]


# Category Selector
class TopicSelectionParser(BaseModel):
    Category:str=Field(description="selected category")
    Reasoning:str=Field(description='Reasoning behind topic selection')
parser = PydanticOutputParser(pydantic_object= TopicSelectionParser)


# Embedding model
#embeddings = HuggingFaceEmbeddings(
#   model_name="sentence-transformers/all-MiniLM-L6-v2"
#)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)


# Initialize Tavily Search Tool
tavily_search_tool = TavilySearch(max_results=5,
    tavily_api_key=TAVILY_API_KEY)
  

# Docs for RAG 
pdf_folder = r"F:\AGENTIC_AI\Agentic_learning\Assignments_AAI\Assignment3\data"
pdf_paths = [os.path.join(pdf_folder,file)
        for file in os.listdir(pdf_folder) if file.endswith(".pdf")]


# Load all documents
documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())


#  Split Text into Chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
uuids = [str(uuid4()) for _ in range(len(docs))]



# Store in vector db
pc = Pinecone(api_key=pinecone_api)
index_name="pinecone-index-rag"
if not pc.has_index(index_name):
    pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws",region="us-east-1")    
)
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index =index,embedding=embeddings)
vector_store.add_documents(docs, ids=uuids)


def retrieve_doc_for_rag(query: str):
   
    pc = Pinecone(api_key=pinecone_api)
    index = pc.Index("pinecone-index-rag")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # hyperparameter 5

    return retriever.invoke(query)




# ------------------ Nodes ------------------

# LLM Node
def llm_node(state: AgentState):
    print("********* LLM NODE ***********")    
    question = state["messages"][0]

    llm_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "You are a knowledgeable assistant and expert on Ontario Lottery. Respond to the user's question in a helpful and informative way. "
            "Write a summary-style paragraph that is well-structured, clear, and around 150 words. "
            "Make sure to highlight **2 to 3 important tips** using markdown-style bolding (e.g., **Legal Age: 18+**, **Use the OLG Website**, etc.). "
            "Avoid bullet points or lists. Provide helpful context and make the response feel complete."),
        ("human", "messages: {question}\n\nWrite the response in paragraph format with bolded tips included, targeting around 150 words.")
    ])

    llm_chain = llm_prompt | model | StrOutputParser()
    response = llm_chain.invoke({"question": question})
    state["source_used"] = "LLM"
    state["messages"] = [response]
    return state





# RAG Node
def rag_node(state: AgentState):

    
    print("********* RAG NODE ***********")
    question=state["messages"][0]
    
    docs= retrieve_doc_for_rag(question)
    
    if not docs:
        return {"messages": ["I couldn't find any relevant information."]}
    
    context = "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)
  
    template = """
You are a legal expert assistant helping users understand official rules and conditions for playing lottery games in Ontario.

Answer the user's question using **only the provided context**, and extract accurate information such as eligibility, age restrictions, game mechanics, and official procedures.

If the context does not contain the necessary details, reply:
"Sorry, I couldn't find an answer in the documents."

---------------------
Context:
{context}
---------------------

Question: {question}

Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    rag_chain = prompt | model 
    
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    final_answer = response.content if hasattr(response, 'content') else str(response)
    print("\nFinal RAG Response:\n", final_answer)
    state["source_used"] = "RAG"
    state["messages"] = [final_answer]
    return state





    
# Web Crawler Node

def extract_latest_olg_links(main_url, count=5):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(main_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.select("a[href*='olg.ca/news-and-media/olg-news']")
        
        links = []
        for a in anchors:
            href = a.get("href")
            if href and href.startswith("https"):
                links.append(href)
            elif href:
                links.append("https://about.olg.ca" + href)
        
        # Remove duplicates and limit count
        return list(dict.fromkeys(links))[:count]
    except Exception as e:
        print(f"Error during link extraction: {e}")
        return []

def fetch_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        article = soup.find("div", class_="content") or soup.find("article") or soup.body
        if article:
            return article.get_text(separator="\n").strip()
        return ""
    except Exception as e:
        print(f"Error fetching article {url}: {e}")
        return ""

def find_most_relevant_doc(query, docs):
    contents = docs
    vectorizer = TfidfVectorizer().fit_transform([query] + contents)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_index = similarities.argmax()
    return best_index

def summarize_text(text, query, model):
    prompt = f"""You are a Canadian lottery news expert.

User query: "{query}"

Here is the content of a recent press release:
\"\"\"{text[:4000]}\"\"\"

Summarize this article in a way that answers the user query.
"""
    try:
        response = model.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        return f"Error during summarization: {e}"
    

def crawler_node(state: dict) -> dict:
    print("************CRAWLER NODE***************")
    
    query = state["messages"][0]  
    source_url = "https://about.olg.ca/news-and-media/olg-news/view-full-list/"
    
    article_links = extract_latest_olg_links(source_url, count=5)
    if not article_links:
        return {**state, "summary": "No articles found.", "source_url": ""}
    
    articles = [fetch_article_text(url) for url in article_links]
    # Filter out empty articles
    articles = [a for a in articles if a.strip()]
    if not articles:
        return {**state, "summary": "Failed to load articles.", "source_url": ""}
    
    best_index = find_most_relevant_doc(query, articles)
    best_text = articles[best_index]
    best_url = article_links[best_index]
    
    summary = summarize_text(best_text, query, model)
    
    return {
        **state,
        "summary": summary,
        "source_url": best_url,
        "source_used": "crawler"
    }





# Supervisor node
def supervisor_node(state: AgentState) -> AgentState:

    print("********* SUPERVISOR NODE ***********")    
    question = state["messages"][-1]    
    
    print(f"Question asked by user: {question}")
    
    template = """
You are a classification agent for a lottery chatbot. Classify the user's query into **only one** of the following categories:

1. **llm** - for general advice, predictions, winning tips, number suggestions, opinions, or how to enroll in lottery games.  
   Example queries:  
   - "How do I win the lottery?"  
   - "Best numbers to play for Lotto Max?"  
   - "Is playing the lottery worth it?"  
   - "Give me tips on how to get enrolled for lottery game."

2. **rag** - for official information, rules, game formats, eligibility, draw days, or ticket guidelines.  
   Example queries:  
   - "How to play Lotto 6/49?"  
   - "Can non-residents buy OLG tickets?"  
   - "What are the prize tiers in Daily Grand?"

3. **web crawler** - for questions involving current or recent events, results, jackpots, or live updates.  
   Example queries:  
   - "What is today's Lotto Max jackpot?"  
   - "Latest winning numbers for Daily Keno"  
   - "When was the last Lotto 6/49 draw?"

User query: {question}

Respond with **only** one of these keywords: llm, rag, or web crawler.
{format_instructions}
"""


    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | model | parser
    
    response = chain.invoke({"question": question})
    print("Parsed response:", response)


    category = response.Category.lower()  # e.g., 'llm', 'rag', 'web crawler'
    
    # Save category in state explicitly
    state["category"] = category
    state["source_used"] = category
    
    # Optionally, keep the messages intact or append the category if you want
    state["messages"].append(category)
    
    return state




# Validation Node
def validation_node(state: dict) -> dict:
    print("************ VALIDATION NODE ***************")

    user_query = state["messages"][0]
    category = state["messages"][-1]  # Should be: "llm", "rag", or "web crawler"

    # Extract the answer based on which node was called
    if category == "web crawler":
        answer = state.get("summary", "")
    elif category == "rag":
        # Assuming rag_node returns a state with the answer inside state["messages"][-2]
        answer = state.get("messages", ["", ""])[-2]
    elif category == "llm":
        # Same structure as rag
        answer = state.get("messages", ["", ""])[-2]
    else:
        print("Unknown category. Cannot validate.")
        state["validation_passed"] = False
        return state

    # Create the validation prompt
    prompt = f"""
You are a validator model. Your job is to determine whether the assistant's response effectively answers the user's question in a helpful and informative way.

---

**User Question**:
{user_query}

**Assistant Response**:
{answer}

---

Return `YES` if all of the following are true:
- The assistant's response is directly relevant to the user's question.
- It provides correct, clear, and useful information.
- It gives at least one actionable step, tip, or informative explanation.
- It is complete enough to give the user a meaningful understanding of the topic.

Return `NO` only if:
- The response is factually wrong, unrelated, or avoids the question.
- It lacks any clear or helpful content (e.g., too vague, too short, or evasive).
- It would confuse or mislead the user rather than help.

---

✅ Examples of `YES`:
Q: How to enroll in lottery?
A: Go to a licensed retailer or OLG.ca, be 18+, select a game, pay, and keep your ticket.

Q: Latest lottery updates?
A: OLG recently announced a new winner of $40M in Lotto Max, and ticket issues have been resolved. [Source]

❌ Examples of `NO`:
Q: What are the lottery rules?
A: I don’t know.

Q: How to play the lottery?
A: Gambling is bad. Don’t do it.

---

Respond with a single word only: YES or NO.
"""
    try:
        response = model.invoke(prompt)
        validation_result = response.content.strip().upper() if hasattr(response, "content") else str(response).strip().upper()
        print(f"Validation model response: {validation_result}")

        validation_passed = validation_result == "YES"
    except Exception as e:
        print(f"Validation error: {e}")
        validation_passed = False

    state["validation_passed"] = validation_passed
    return state





# Routing Logic 
def router(state: AgentState) -> dict:
    print("********* ROUTER NODE ***********")
    last_message = state["messages"][-1]
    if isinstance(last_message, dict) and "Category" in last_message:
        category = last_message["Category"].lower()
    elif isinstance(last_message, str):
        category = last_message.lower()
    else:
        category = "llm"  # default fallback    
    
    state["user_query"] = state["messages"][0]
    state["route"] = category
    state["source_used"] = category

    return state

# Validation Routing
def validate_and_reroute(state: dict):
    return END if state.get("validation_passed") else "supervisor"



# ------------------ Workflow Setup ------------------

workflow = StateGraph(AgentState)

# 1. Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("router", router)
workflow.add_node("llm_node", llm_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("crawler_node", crawler_node)
workflow.add_node("validation_node", validation_node)

# 2. Entry Point
workflow.set_entry_point("supervisor")

# 3. Supervisor routes to router
workflow.add_edge("supervisor", "router")

# 4. Router routes based on decision
def route_decision(state: AgentState) -> str:
    return state.get("route", "llm")

workflow.add_conditional_edges("router", route_decision, {
    "llm": "llm_node",
    "rag": "rag_node",
    "web crawler": "crawler_node"
})

# 5. After any path, validate
#workflow.add_edge("llm_node", "validation_node")
#workflow.add_edge("rag_node", "validation_node")
#workflow.add_edge("crawler_node", "validation_node")

# 6. Validation routes to END or restart
#workflow.add_conditional_edges("validation_node", validate_and_reroute, {
    #    "supervisor": "supervisor",
    #    END: END
    #})

workflow.add_edge("llm_node", END)
workflow.add_edge("rag_node", END)
workflow.add_edge("crawler_node", END)

# 7. Compile
app = workflow.compile()

    # === Streamlit UI ===
st.title("Lottery Tips Assistant with LangGraph Workflow")

user_input = st.text_area("Ask your question about Ontario Lottery:", height=120)

if st.button("Get Tips"):
    if user_input.strip() == "":
        st.warning("Please enter your question!")
    else:
        with st.spinner("Processing..."):
            state = {"messages": [user_input.strip()]}
            final_state = app.invoke(state)
            output = final_state.get("messages")[-1]
            source = final_state.get("source_used", "Unknown").upper()

            # Present the output 
            st.markdown("### Response:")
            st.markdown(f"<div style='padding:10px; border-left: 5px solid #007bff; background-color: #f0f8ff;'>{output}</div>", unsafe_allow_html=True)

            #st.markdown(f"**Response Source:** 🛠️ {source}")

            # Optionally show source URL and summary if present
            if "source_url" in final_state:
                st.markdown(f"**Source URL:** [Link]({final_state['source_url']})")
            if "summary" in final_state and final_state["summary"]:
                st.markdown(f"**Summary:** {final_state['summary']}")