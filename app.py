import streamlit as st
from streamlit_lottie import st_lottie
import requests

from streamlit_gallery import components
from streamlit_gallery.utils.page import page_group
from tweak_customer_churn import TweakCustomerChurn

@st.cache_data
def load_lottie_url(url: str):
    
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():

    # Custom CSS
    styles = """
        <style>
            body {
                background-color: #FAF3E0; 
                font-family: "Arial", sans-serif;
            }
            
            h1 {
                font-family: "Georgia", monospace; 
                color: #3E2723;
            }
            
            .stButton>button {
                background-color: #575735;
                color: white !important;
            }
        </style>
    """
    
    st.markdown(styles, unsafe_allow_html=True)
    
    #Lottie
    lottie_url = "https://lottie.host/80aa7bd9-c452-4ce6-a92b-67db63ad9ffc/b62iBOixT4.json"  # Sample URL, replace with your desired animation
    lottie_animation = load_lottie_url(lottie_url)
    st_lottie(lottie_animation, speed=1, width=250, height=250)
    
    st.title('Telkomsel Data Challenge Churn Prediction')
    st.markdown("""Due to resource constraint provided by Streamlit Sharing, only permitted users are allowed access. Please note that the app interface is not flawless, occasional state rollback may occur. Nevertheless, the app serves its purpose of demonstrating the model's performance.""")
    

    # password_guess = st.text_input('What is the Password?') 
    # if password_guess != st.secrets["password"]:
    #     st.stop()

    # Sidebar for navigation
    page = page_group("p")

    with st.sidebar:
        st.title("🕵️‍♂️ Detective Gallery")
        st.caption("where Storytelling meets Modelling")
        st.write("")
        st.markdown('Made by <a href="https://www.linkedin.com/in/andreaslukita7/">Andreas Lukita</a>, <a href="https://www.linkedin.com/in/ruben-daniel/">Ruben Daniel</a>, and <a href="https://www.linkedin.com/in/ronaldo-w-50783b171/">Ronaldo Wijaya</a>', unsafe_allow_html=True)

        with st.expander("⏳ COMPONENTS", True):
            page.item("Introduction", components.show_introduction, default=True)
            page.item("Prediction and Modelling⭐", components.show_prediction)
            page.item("Model Visualisation", components.show_modelviz)

    page.show()
    
if __name__ == "__main__":
    main()