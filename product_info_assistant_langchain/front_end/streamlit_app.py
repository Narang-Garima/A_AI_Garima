# front_end/streamlit_app.py

import streamlit as st
import sys
import os
from typing import List

# Add root directory to Python path so imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)


from pipeline.chaining import create_chain, get_product_info


def main():
    st.title("Product Info Assistant")

    # Model selection
    model_list = ["gemma2-9b-it", "llama3-70b-8192"]
    model_name = st.selectbox("Select model", model_list)
    try:
        chain = create_chain(model_name)
    except Exception as e:
        st.error(f"Error while creating model chain: {e}")
        st.stop()

    product = st.text_input("Enter product name")
    
    if product and st.button("Get Products Info"):
        print(f"Calling get_product_info with chain: {chain} and product: '{product}'")
        response = get_product_info(chain, product)
        if response:
            st.markdown("---")
            st.subheader("🧾 Product Details:")
            st.write(f"**Product_name:** {response.product_name}")
            st.write(f"**Product Details:** {response.product_details}")
            st.write(f"**Tentative Price (USD):** ${response.tentative_price_usd}")
            st.markdown("---")
            st.success("Product details fetched successfully!")
        else:
            st.warning("No product info available.")

if __name__ == "__main__":
    main()
