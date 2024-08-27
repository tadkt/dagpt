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

def main():
    # Set up Streamlit interface
    st.set_page_config(
        page_title="Smart Data Analysis Tool",
        page_icon="📊",
        layout="centered"
    )

    # Load LLMs model
    llm = load_llm(model_name = MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME} ###")
    # Upload CSV file

    # Install chat history

    # Read csv file

    # Create data analysis agent to query with our data

    # Input query and process query

    # Display chat history


    pass

if __name__ == "__main__":
    main()
