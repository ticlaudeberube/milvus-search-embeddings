from tqdm import tqdm
from ollama import chat, ChatResponse
import sys, os
from termcolor import colored, cprint

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.MilvusUtils import MilvusUtils

client = MilvusUtils.get_client()
collection_name = os.getenv("OLLAMA_COLLECTION_NAME") or "demo_collection"

# prepare prompt
question = "How is data stored in milvus?"

#start seaeching
cprint('\nPreparing...\n', 'green', attrs=['blink'])
search_res = client.search(
    collection_name=collection_name,
    data=[
       MilvusUtils.embed_text_ollama(question)
    ],
    limit=3,  # Return top 3 results
    search_params={"metric_type": "COSINE", "params": {"radius": 0.4, "range_filter": 0.7} },  # Cosine similarity
    output_fields=["text"],  # Return the text field
)

#print(search_res)

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
#print(json.dumps(retrieved_lines_with_distances, indent=4))

context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
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
context = "\n".join(
    [line_with_distance[0] for line_with_distance in tqdm(retrieved_lines_with_distances, desc="Processing results")]
)
def rag_query():
    print('\nUser prompt:\n'+ USER_PROMPT)
    cprint(f"Searching... {question}\n", 'green', attrs=['blink'])
    llm_model = os.getenv("OLLAMA_LLM_MODEL")
    response: ChatResponse = chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print(response["message"]["content"])
    return

rag_query()
