import os
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from data_converter import data_converter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
def load_env_variables():
    try:
        load_dotenv(dotenv_path='./.env')
        logger.info("Environment variables loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        raise  # Re-raise exception after logging

# Validate necessary environment variables
def validate_env_variables():
    required_vars = [
        "HF_TOKEN", "OPENAI_API_KEY", "ASTRA_DB_API_ENDPOINT",
        "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("All required environment variables are set.")

# Extracted method to initialize embeddings
def initialize_embeddings():
    embedding_hf = None
    embedding_oai = None

    # Try loading HuggingFace embeddings
    try:
        embedding_hf = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("HF_TOKEN"), model_name="BAAI/bge-base-en-v1.5")
        logger.info("HuggingFace embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing HuggingFace embeddings: {e}")

    # Try loading OpenAI embeddings
    try:
        embedding_oai = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAI embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI embeddings: {e}")

    if embedding_oai is None:
        raise ValueError("Embedding model not properly initialized.")

    return embedding_hf, embedding_oai

# Create and configure the vector store, including embeddings and data ingestion
def create_vector_store():
    # Ensure environment variables are loaded before using them
    load_env_variables()
    validate_env_variables()

    # Initialize embeddings
    embedding_hf, embedding_oai = initialize_embeddings()

    try:
        # Create AstraDB Vector Store
        vstore = AstraDBVectorStore(
            collection_name="ecomm",
            embedding=embedding_oai,
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            namespace=os.getenv("ASTRA_DB_KEYSPACE")
        )
        logger.info("Vector store created successfully.")

        # Ingest data into the vector store
        docs = data_converter()  # Ensure this returns a list of Document objects
        if not docs:
            logger.warning("No documents returned by data_converter.")

        # Add documents to the vector store
        insert_ids = vstore.add_documents(docs)
        logger.info(f"Inserted {len(insert_ids)} documents into vector store.")
        
        return vstore, insert_ids
    
    except Exception as e:
        logger.error(f"Error during vector store creation or data ingestion: {e}")
        raise  # Re-raise exception after logging

# Perform similarity search on the vector store
def search_vector_store(vstore, query):
    try:
        results = vstore.similarity_search(query)
        for res in results:
            logger.info(f"\n{res.page_content} [{res.metadata}]")
    except Exception as e:
        logger.error(f"Error during vector store search: {e}")
        raise  # Re-raise exception after logging

# Main execution logic
def main():
    try:
        # Create the vector store and ingest data
        vstore, insert_ids = create_vector_store()

        # Perform a search on the vector store if documents were inserted
        if insert_ids:
            search_vector_store(vstore, "Can you tell me the low budget sound basshead?")
        else:
            logger.info("No documents were inserted.")
    
    except Exception as e:
        logger.error(f"An error occurred during the execution: {e}")

# Entry point for the script
if __name__ == "__main__":
    main()
