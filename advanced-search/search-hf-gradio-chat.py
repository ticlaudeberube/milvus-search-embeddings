from tqdm import tqdm
import sys, os
from termcolor import colored, cprint
import gradio as gr
from huggingface_hub import InferenceClient

# Add HF token for model access
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Please set HUGGINGFACE_TOKEN environment variable")

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, timeout=120)

sys.path.insert(1, './utils')
from MilvusUtils import MilvusUtils

client = MilvusUtils.get_client()
collection_name = os.getenv("MILVUS_HF_COLLECTION_NAME") or "demo_collection"

def embed_text(text):
    response = MilvusUtils.embed_text_hf(text)
    print(response[0])
    return response

# TODO: implement history
def rag_query(question, history=[]):
    limit = 10
    cprint(f'\nRetreiving context: limit ({limit}) documents...\n', 'green', attrs=['blink'])

    #start searching
    search_res = client.search(
        collection_name=collection_name,
        data=[
            embed_text(question)
        ],
        limit=limit,
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # Sort by distance
    context = "\n".join(
        [line_with_distance[0] for line_with_distance in tqdm(retrieved_lines_with_distances, desc="Processing results")]
    )

    cprint('\nSearching...\n', 'green', attrs=['blink'])
    PROMPT = f"""
    Human: You are an AI assistant. You are able to find answers to the questions from 
    the contextual passage snippets provided.
    Use the following pieces of information enclosed in <context> tags to provide an answer 
    to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    cprint('\nSearching...\n', 'green', attrs=['blink'])
    prompt = PROMPT.format(context=context, question=question)
    response = llm_client.text_generation(
        prompt,
        max_new_tokens=1000,
    ).strip()
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
)

if __name__ == "__main__":
    chat.launch()
