## 🛠️ LLM-Powered Product Info Assistant using Pydantic & JSON Parser

This assistant fetches structured product details like name, description, and price in USD using a Large Language Model (LLM). The output is parsed into a JSON object using Pydantic for type safety.

### 🚀 Workflow Overview

1. **Environment Setup**:
   - Store and load your API keys securely using environment variables or a `.env` file.
   - Example:
     ```python
     import os
     from dotenv import load_dotenv
     load_dotenv()
     api_key = os.getenv("OPENAI_API_KEY")
     ```

2. **Model Initialization**:
   - Instantiate the LLM model (e.g., OpenAI) using the API key.
   - This model will generate natural language responses.

3. **Prompt Template Definition**:
   - Define a prompt template instructing the model to act as an **E-commerce Product Expert**.
   - The template should clearly request the response in a **JSON format** with specific fields.

4. **Define Pydantic Model**:
   - Use Pydantic to define a schema for the expected structured response.
     ```python
     from pydantic import BaseModel

     class ProductInfo(BaseModel):
         name: str
         description: str
         price_usd: float
     ```

5. **Set Up Output Parser**:
   - Use `JsonOutputParser` or equivalent to parse model output into the defined `ProductInfo` model.
     ```python
     from langchain.output_parsers import JsonOutputParser
     parser = JsonOutputParser(pydantic_object=ProductInfo)
     ```

6. **Chain the Components**:
   - Create a chain where:
     - The **PromptTemplate** feeds into the **LLM**
     - The LLM response is passed into the **Parser**
     - The final output is a structured `ProductInfo` object

7. **Invoke the Chain**:
   - Provide a user query like:
     > "Tell me about the Apple iPhone 15"
   - Receive a structured response:
     ```json
     {
       "name": "Apple iPhone 15",
       "description": "The latest iPhone model with A16 chip, 5G support, and advanced camera system.",
       "price_usd": 799.0
     }
     ```

---

### 💡 Technologies Used
- Python
- OpenAI API
- LangChain
- Pydantic
- JSON Parser


conda activate C:\Users\KrishnaDasaNuDasi\miniconda3\envs\.agentic_base

streamlit run Assignments_AAI/product_info_assistant_langchain/front_end/streamlit_app.py


docker build -t product_info_assistant_langchain .
docker run -p 8501:8501 --env-file .env product_info_assistant_langchain
http://localhost:8501
