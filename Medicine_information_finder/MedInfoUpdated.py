# ------------------ Install Required Packages ------------------
import os 
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-groq"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "wikipedia", "wikipedia-api", "beautifulsoup4"])

# ------------------ Imports ------------------
import streamlit as st
import wikipedia
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ------------------ API Keys ------------------
load_dotenv(override=True)
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ------------------ Model Setup ------------------
model = ChatGroq(model="gemma2-9b-it", temperature=0.7)

class MedicineInfo(BaseModel):
    name: str = Field(..., description="Name of the medicine")
    composition: List[str] = Field(..., description="List of active ingredients")
    brand_alternatives: Optional[List[str]] = Field(None)
    price_inUSD: Optional[float] = Field(None)
    uses: Optional[List[str]] = Field(None)
    dosage: Optional[str] = Field(None)
    side_effects: Optional[List[str]] = Field(None)
    contraindications: Optional[List[str]] = Field(None)
    prescription_required: Optional[bool] = Field(None)

parser = JsonOutputParser(pydantic_object=MedicineInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert Pharmacist. Refer to trusted sources like Drugs.com. "
     "Also act as an e-commerce expert to estimate price in USD."),
    ("user", "#Format: {format_instructions}\n\n#Question :{question}")
])
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser

# ------------------ Function to Fetch Image ------------------
def fetch_medicine_image(medicine_name):
    try:
        page = wikipedia.page(medicine_name)
        html = page.html()
        soup = BeautifulSoup(html, 'html.parser')
        img_tag = soup.find('img')
        if img_tag:
            return 'https:' + img_tag['src']
    except Exception:
        return None

# ------------------ Streamlit UI Setup ------------------
st.set_page_config(page_title="💊 Medicine Info Finder", page_icon="💊", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .main {
        background-color: black;
        padding: 2rem;
        border-radius: 10px;
    }
    .title-text {
        font-size: 32px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }
    .subtitle-text {
        font-size: 18px;
        color: #cccccc;
    }
    .stTextInput input {
        background-color: white;
        color: black;
    }
    .stButton button {
        color: red;
        border: 1px solid red;
        background-color: transparent;
    }
    h3, h4, h2, h1 {
        color: yellow;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="main">
        <h1 class="title-text">💊 Medicine Info Finder</h1>
        <p class="subtitle-text">Get instant, reliable information on any medicine.</p>
    </div>
""", unsafe_allow_html=True)

# Input + Button
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("", placeholder="🔍 Enter medicine name (e.g., Tylenol)...")
with col2:
    st.write("") 
    st.write("")
    search_button = st.button("Search", use_container_width=True)

# On Search
if search_button and user_query:
    with st.spinner("Fetching information..."):
        try:
            response_data = chain.invoke({"question": f"Tell me about {user_query}"})
            response = MedicineInfo(**response_data)

            st.subheader("🔍 Medicine Details")

            image_url = fetch_medicine_image(user_query)
            if image_url:
                st.image(image_url, caption=f"{response.name} - representative image", width=200)

            st.markdown(f"**Name:** {response.name}")
            st.markdown(f"**Composition:** {', '.join(response.composition or [])}")
            
            price = f"${response.price_inUSD:.2f}" if response.price_inUSD is not None else "Not Available"
            st.markdown(f"**Price in USD:** {price}")
            st.markdown(f"**Prescription Required:** {'Yes' if response.prescription_required else 'No'}")

            with st.expander("📋 Usage"):
                st.write(', '.join(response.uses or []))

            with st.expander("⚠️ Side Effects"):
                st.write(', '.join(response.side_effects or []))

            with st.expander("🔄 Brand Alternatives"):
                st.write(', '.join(response.brand_alternatives or []))

            with st.expander("💡 Dosage"):
                st.write(response.dosage or 'N/A')

            with st.expander("🚫 Contraindications"):
                st.write(', '.join(response.contraindications or []))

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.warning("Please enter a medicine name before searching.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; font-size: 14px; color: grey;'>
        💡 Powered by <b>LangChain</b> + <b>Groq</b> | Developed by <i>Garima</i><br>
        🌐 <a href='https://www.linkedin.com/in/garima-narang-62145b57/' target='_blank' style='color: #0e76a8;'>Connect with me on LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
