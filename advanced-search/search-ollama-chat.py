from tqdm import tqdm
from ollama import chat, ChatResponse
import sys, os
from termcolor import colored, cprint


sys.path.insert(1, './utils')
from MilvusUtils import MilvusUtils

client = MilvusUtils.get_client()
collection_name = os.getenv("MILVUS_OLLAMA_COLLECTION_NAME") or "demo_collection"


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
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
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
    response: ChatResponse = chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print(response["message"]["content"])
    return

rag_query()
