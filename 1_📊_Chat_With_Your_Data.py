import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from src.models.llms import load_llm
from src.logger.base import BaseLogger
# Load environment variables
load_dotenv()
logger = BaseLogger()
MODEL_NAME = 'gpt-3.5-turbo'


def process_query(da_agent, query):
    pass

def main():
    # Set up Streamlit interface
    st.set_page_config(
        page_title="📊 Smart Data Analysis Tool",
        page_icon="📊",
        layout="centered"
    )
    st.header("# 📊 Smart Data Analysis Tool")
    st.write("Welcome to our data analysis tool. This tool can assist your data analysis test. Please enjoy :)")

    # Load LLMs model
    llm = load_llm(model_name = MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME} ###")
    # Upload CSV file
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your csv file here", type="csv")
    # Install chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Read csv file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Your uploaded data: ", st.session_state.df.head())
    # Create data analysis agent to query with our data
    da_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state.df,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=True,
        return_intermediate_steps=True
    )
    logger.info("### Successfully loaded data analysis agent")
    # Input query and process query
    query = st.text_input("Enter your question: ")
    
    if st.button("Run query"):
        with st.spinner("Processing..."):
            process_query(da_agent, query)
    # Display chat history


    pass

if __name__ == "__main__":
    main()
