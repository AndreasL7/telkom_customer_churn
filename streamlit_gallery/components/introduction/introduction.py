import streamlit as st
import gc

def main():
    
    gc.enable()
    
    st.subheader("Welcome, Detective!")
    
    st.write("""
    In the rapidly evolving telecommunications landscape of Indonesia, a cutthroat competition amongst industry stalwarts shapes the market's trajectory. Spearheaded by key players such as Telkom Indonesia, Indosat Ooredoo, XL Axiata, Smartfren Telecom, and Tri Indonesia, the sector exhibits a dynamic interplay of market forces striving for a larger share of the connectivity domain​.
    """)
    
    image_url = "https://stl.tech/wp-content/uploads/2022/08/future-of-telcos-1.jpg"
    
    st.markdown(f'<a href="{image_url}"><img src="{image_url}" alt="description" width="700"/></a>', unsafe_allow_html=True)
    
    # r
        
    st.write("""
    
    Navigate to the **Prediction and Modelling** page to understand how our model works. 
    
    Grab your coffee and enjoy the investigation ahead! ☕️
    """)
    
    gc.collect()

if __name__ == "__main__":
    main()