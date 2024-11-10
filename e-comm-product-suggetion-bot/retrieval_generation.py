import os
import logging
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from data_ingestion import data_ingestion  

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables and set up logging
def load_environment():
    try:
        load_dotenv(dotenv_path='./.env')
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        logger.info("Environment variables loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading environment variables: {e}")
        raise

# Initialize the ChatGroq model
def initialize_model():
    try:
        model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
        logger.info("Model initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

# Initialize session history storage
def get_session_history(session_id: str, store: dict) -> BaseChatMessageHistory:
    try:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise

# Create a retriever for documents from the vector store
def create_retriever(vstore):
    try:
        retriever = vstore.as_retriever(search_kwargs={"k": 3})
        logger.info("Retriever created successfully.")
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise

# Create the question answering (QA) prompt template
def create_qa_prompt():
    try:
        PRODUCT_BOT_TEMPLATE = """
        Your ecommercebot bot is an expert in product recommendations and customer queries.
        It analyzes product titles and reviews to provide accurate and helpful responses.
        Ensure your answers are relevant to the product context and refrain from straying off-topic.
        Your responses should be concise and informative.

        CONTEXT:
        {context}

        QUESTION: {input}

        YOUR ANSWER:
        """
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", PRODUCT_BOT_TEMPLATE), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
        )
        logger.info("QA prompt created successfully.")
        return qa_prompt
    except Exception as e:
        logger.error(f"Error creating QA prompt: {e}")
        raise

# Create the conversational retrieval-augmented generation (RAG) chain
def create_conversational_chain(vstore, model, store):
    try:
        retriever = create_retriever(vstore)

        # Define retriever prompt
        retriever_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )

        # Create prompt template for contextualizing the question
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", retriever_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
        )

        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

        # Create the QA chain
        qa_prompt = create_qa_prompt()
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

        # Combine with retrieval chain to form the full RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        logger.info("Conversational chain created successfully.")
        return RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: get_session_history(session_id, store),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    except Exception as e:
        logger.error(f"Error creating conversational chain: {e}")
        raise

# Main function that integrates data ingestion, creation of the vector store, and conversational logic
def main():
    try:
        # Load environment variables and initialize model
        load_environment()
        model = initialize_model()

        # Ingest data and create the vector store
        vstore, insert_ids = data_ingestion()

        # Proceed if documents were inserted
        if insert_ids:
            store = {}
            conversational_rag_chain = create_conversational_chain(vstore, model, store)

            # Perform a query
            answer = conversational_rag_chain.invoke(
                {"input": "Can you tell me the best bluetooth buds?"},
                config={"configurable": {"session_id": "dhruv"}}
            )["answer"]
            logger.info(f"Response: {answer}")

            # Query for the previous question
            answer1 = conversational_rag_chain.invoke(
                {"input": "What is my previous question?"},
                config={"configurable": {"session_id": "dhruv"}}
            )["answer"]
            logger.info(f"Response: {answer1}")
        else:
            logger.warning("No documents were inserted into the vector store.")

    except Exception as e:
        logger.error(f"An error occurred during the execution: {e}")

# Entry point for the script
if __name__ == "__main__":
    main()
