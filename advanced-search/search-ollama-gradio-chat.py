from tqdm import tqdm
from ollama import chat, ChatResponse
import sys, os
from termcolor import colored, cprint
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils

client = MilvusUtils.get_client()
collection_name = os.getenv("MILVUS_OLLAMA_COLLECTION_NAME") or "demo_collection"

def embed_text(text):
    response = MilvusUtils.embed_text_ollama(text)
    print(response[0])
    return response

def rag_query(question):
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

    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    cprint('\nSearching...\n', 'green', attrs=['blink'])
    response: ChatResponse = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],  
    )
    print(response["message"]["content"])
    cprint('\nDone! \n', 'green', attrs=['blink'])
    return response["message"]["content"]

defaultInference="How is data stored in milvus?"
# Create Gradio interface
interface = gr.Interface(
    fn=rag_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here...", value=defaultInference ),
    outputs="text",
    title="Query Milvus Docs",
    description="Ask questions about Milvus and get answers from the knowledge base"
)

# Launch the interface
interface.launch()
# embed_text(defaultInference)
