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
from data_ingestion import data_ingestion  # Assuming this import is correct

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EcommerceBot:
    def __init__(self):
        self.store = {}
        self.model = None
        self.vstore = None
        self.conversational_rag_chain = None
        self.load_environment()

    def load_environment(self):
        """Load environment variables from .env file."""
        try:
            load_dotenv()
            os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
            logger.info("Environment variables loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            raise

    def initialize_model(self):
        """Initialize the ChatGroq model."""
        try:
            self.model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.5)
            logger.info("Model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Fetch or create session history."""
        try:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            raise

    def create_retriever(self):
        """Create the document retriever."""
        try:
            retriever = self.vstore.as_retriever(search_kwargs={"k": 3})
            logger.info("Retriever created successfully.")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise

    def create_qa_prompt(self):
        """Create the QA prompt template."""
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

    def create_conversational_chain(self):
        """Create the full conversational retrieval-augmented generation chain."""
        try:
            retriever = self.create_retriever()

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
            history_aware_retriever = create_history_aware_retriever(self.model, retriever, contextualize_q_prompt)

            # Create the QA chain
            qa_prompt = self.create_qa_prompt()
            question_answer_chain = create_stuff_documents_chain(self.model, qa_prompt)

            # Combine with retrieval chain to form the full RAG chain
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            logger.info("Conversational chain created successfully.")
            self.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: self.get_session_history(session_id),
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        except Exception as e:
            logger.error(f"Error creating conversational chain: {e}")
            raise

    def ingest_data(self):
        """Ingest data into the vector store."""
        try:
            self.vstore, insert_ids = data_ingestion()
            if not insert_ids:
                logger.warning("No documents were inserted into the vector store.")
            else:
                logger.info(f"Inserted {len(insert_ids)} documents into the vector store.")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise

    def process_query(self, query: str, session_id: str):
        """Process a user query using the conversational chain."""
        try:
            if not self.conversational_rag_chain:
                logger.error("Conversational RAG chain is not initialized.")
                raise ValueError("Conversational RAG chain is not initialized.")
            
            answer = self.conversational_rag_chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )["answer"]
            logger.info(f"Response: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise



