import streamlit as st
from SalesGPT import SalesGPT  
from SalesGPT import SalesConversationChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message

"""Install Required Packages:
Make sure you have Streamlit and any other required packages installed. You can install them using:

pip install streamlit
Run the Streamlit App:
Open a terminal and run the Streamlit app script:

bash
Copy code
streamlit run pp.py
This will launch a local development server, and the Streamlit app will be accessible in your web browser.

Interact with the App:
Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501). You can interact with the Streamlit app by entering user inputs and clicking the "Submit" button to see the agent's responses and the conversation history."""

llm = ChatOpenAI(temperature=0.7)


json_data = '''
{
  "inventory": [
    {
      "product_id": "RO1001",
      "product_name": "AquaPure Home",
      "description": "5-stage RO system with UV purification",
      "price": 199.99,
      "currency": "USD",
      "in_stock": 150,
      "replacement_filters_available": true
    },
    {
      "product_id": "RO1002",
      "product_name": "ClearFlow Office",
      "description": "8-stage RO system for high-capacity use",
      "price": 399.99,
      "currency": "USD",
      "in_stock": 75,
      "replacement_filters_available": true
    },
    {
      "product_id": "RO1003",
      "product_name": "HydroHealth Travel",
      "description": "Portable RO purifier for travelers",
      "price": 99.99,
      "currency": "USD",
      "in_stock": 0,
      "replacement_filters_available": false
    },
    {
      "product_id": "RO1004",
      "product_name": "SpringFalls Countertop",
      "description": "Compact RO system with smart monitoring",
      "price": 249.99,
      "currency": "USD",
      "in_stock": 25,
      "replacement_filters_available": true
    }
  ]
}
'''

# Function to initialize the SalesGPT agent
def initialize_sales_agent():
    config = dict(
    salesperson_name = "Ibrahim",
    salesperson_role = "Business Development Representative",
    company_name = "GPT RO purifiers",
    company_business = "GPT RO sell premuim and high quality RO purifiers.",
    company_values = "Our mission is to provide clean and healthy water with minimum cost.",
    conversation_purpose = "find out customer needs and potential solutions for purchasing products.",
    conversation_history = [],
    conversation_type = "chat",
    json_data = json_data,
)
    return SalesGPT.from_llm(llm, verbose=False, **config)


# Load the SalesGPT model
sales_gpt = initialize_sales_agent()

# Streamlit app
st.title("Sales Chatbot")

# Initialize conversation stage and history
sales_gpt.seed_agent()

# Sidebar for user input
user_input = st.text_input("You:", key="user_input")
if user_input:
    # Process user input
    sales_gpt.human_step(user_input)

    # Determine the conversation stage
    sales_gpt.determine_conversation_stage()

    # Run the chatbot to generate a response
    sales_gpt.step()
    sales_gpt.extract_and_save_customer_interest()
    # Display the conversation
    st.markdown(f"**You:** {user_input}", unsafe_allow_html=True)
    st.markdown(f"**{sales_gpt.salesperson_name}:** {sales_gpt.conversation_history[-1].rstrip('<END_OF_TURN>')}", unsafe_allow_html=True)
