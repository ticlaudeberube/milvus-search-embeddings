import os
import sys
import logging
from typing import List, Optional, Tuple
from tqdm import tqdm
import gradio as gr # type: ignore
from huggingface_hub import InferenceClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.utils import MilvusClient

# Environment variable for HuggingFace token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN environment variable")

# Parameterize model repo and collection name
repo_id = os.getenv("HF_REPO_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
collection_name = os.getenv("MILVUS_HF_COLLECTION_NAME", "demo_collection")

try:
    llm_client = InferenceClient(model=repo_id, token=hf_token, timeout=120)
except Exception as e:
    logger.error(f"Failed to initialize InferenceClient: {e}")
    raise

try:
    client = MilvusClient.get_client()
except Exception as e:
    logger.error(f"Failed to get Milvus client: {e}")
    raise

def embed_text(text: str) -> List[float]:
    """
    Embed text using MilvusClient' HuggingFace embedding method.
    Args:
        text (str): The text to embed.
    Returns:
        List[float]: The embedding vector.
    """
    try:
        response = MilvusClient.embed_text_hf(text)
        logger.debug(f"Embedding: {response[0]}")
        return response
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []

def rag_query(question: str, history: List = None) -> str:
    """
    Perform a RAG query: retrieve context from Milvus and generate an answer using an LLM.
    Args:
        question (str): The user's question.
        history (List): Conversation history from Gradio.
    Returns:
        str: The answer from the LLM.
    """
    # Initialize history if None
    if history is None:
        history = []
    
    # Check if question is empty
    if not question.strip():
        return "Please ask a question."
        
    limit = 10
    logger.info(f'Retrieving context: limit ({limit}) documents...')
    
    # Get embeddings for the question
    embeddings = embed_text(question)
    if not embeddings:
        return "Failed to create embeddings for your question."
    
    try:
        search_res = client.search(
            collection_name=collection_name,
            data=[embeddings],
            limit=limit,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )
    except Exception as e:
        logger.error(f"Milvus search failed: {e}")
        return "Error retrieving context from Milvus."

    # Check if search results are empty
    if not search_res or not search_res[0]:
        return "No relevant information found in the database."

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    # Format conversation history
    conversation_context = ""
    if history:
        # Use the last 3 exchanges for context
        recent_history = history[-3:] if len(history) > 3 else history
        for exchange in recent_history:
            if len(exchange) == 2:  # Make sure we have both user and assistant messages
                user_msg, assistant_msg = exchange
                conversation_context += f"Human: {user_msg}\nAssistant: {assistant_msg}\n\n"

    prompt = f"""
    Human: You are an AI assistant. You are able to find answers to the questions from 
    the contextual passage snippets provided.
    Use the following pieces of information enclosed in <context> tags to provide an answer 
    to the question enclosed in <question> tags.
    
    <context>
    {context}
    </context>
    
    <conversation_history>
    {conversation_context}
    </conversation_history>
    
    <question>
    {question}
    </question>
    
    When answering, consider both the context and the conversation history if relevant.
    """
    logger.info('Querying LLM...')
    try:
        response = llm_client.text_generation(
            prompt,
            max_new_tokens=1000,
        ).strip()
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "Error generating answer from LLM."
    logger.info('Done!')
    return response

chat = gr.ChatInterface(
    rag_query,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
    title="RAG with Milvus and HuggingFace",
    description="""
    This is a demo of a Retrieval-Augmented Generation (RAG) system using Milvus and HuggingFace.\n\n"
    "You can ask questions and get answers based on the context retrieved from the Milvus database.\n\n"
    "- The model and collection can be configured via environment variables.\n"
    "- Please do not enter sensitive information.\n"
    """,
)

if __name__ == "__main__":
    chat.launch()
