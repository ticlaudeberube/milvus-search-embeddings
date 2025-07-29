
import os, sys
from termcolor import colored, cprint
import gradio as gr
from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils

# Add HF token for model access
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN environment variable")

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, token=hf_token, timeout=120)

client = MilvusUtils.get_client()
collection_name = os.getenv("HF_COLLECTION_NAME") or "demo_collection"

def embed_text(text: str):
    response = MilvusUtils.embed_text_hf([text])
    return response[0]  # Return the first embedding vector

# TODO: implement history
def rag_query(question: str, history=[]):
    limit = 10
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
        return "Error retrieving context from Milvus."

    #start searching
    search_res = client.search(
        collection_name=collection_name,
        data=[embed_text(question)],
        limit=limit,
        search_params={"metric_type": "COSINE", "params": {"radius": 0.4, "range_filter": 0.7} },
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    history_context = "\n".join([f"Human: {h[0]['text']}\nAssistant: {h[1]['text']}" for h in history[-3:]])
    
    cprint('\nSearching...\n', 'green', attrs=['blink'])
    prompt = f"""
    Human: You are an AI assistant. Use the context and conversation history to answer questions.
    
    Previous conversation:
    {history_context}
    
    Context:
    {context}
    
    Question: {question}
    """
    
    messages = [{"role": "user", "content": prompt}]
    response = llm_client.chat_completion(messages, max_tokens=1000).choices[0].message.content
    print(response)
    cprint('\nDone! \n', 'green', attrs=['blink'])
    return response

chat = gr.ChatInterface(
    rag_query,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
    title="RAG with Milvus and HuggingFace",
    description="This is a demo of a Retrieval-Augmented Generation (RAG) system using Milvus and HuggingFace. "
                "You can ask questions and get answers based on the context retrieved from the Milvus database.",
    examples=["What is Milvus"]
)

if __name__ == "__main__":
    chat.launch()
