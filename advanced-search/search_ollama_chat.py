import os
from tqdm import tqdm
from ollama import chat, ChatResponse
from termcolor import cprint

from dotenv import load_dotenv
load_dotenv()

from core import MilvusUtils

client = MilvusUtils.get_client()
collection_name = os.getenv("OLLAMA_COLLECTION_NAME") or "demo_collection"

# Check if collection exists
if not MilvusUtils.has_collection(collection_name):
    cprint(f"\nCollection '{collection_name}' not found!", 'red')
    cprint("Please load data first using one of these scripts:", 'yellow')
    cprint("  python document-loaders/load_milvus_docs_ollama.py", 'cyan')
    exit(1)

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
    llm_model = os.getenv("OLLAMA_LLM_MODEL", '')
    response: ChatResponse = chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
    ) # type: ignore
    print(response["message"]["content"])
    return

rag_query()
