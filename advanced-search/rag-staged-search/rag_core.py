import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from core.MilvusUtils import MilvusUtils

def needs_retrieval(llm, question):
    """Stage 1: Check if question needs document retrieval"""
    classification_prompt = f"""
        Human: Answer "YES" if this question is about Milvus database, vector databases, or related technical topics that would benefit from documentation lookup. Answer "NO" for instructions, greetings, personal introductions, or general conversation.
        
        Question: {question}

        Assistant:"""
    
    response = llm.invoke(classification_prompt)
    return "YES" in response.upper()

def rag_query_with_retrieval(client, llm, collection_name, question, chat_history):
    """Stage 2: Full RAG with retrieval"""
    search_res = client.search(
        collection_name=collection_name,
        data=[MilvusUtils.embed_text_ollama(question)],
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"radius": 0.4, "range_filter": 0.7}},
        output_fields=["text"]
    )
    
    context = "\n\n".join([res["entity"]["text"] for res in search_res[0]])
    history_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history[-3:]])
    
    prompt = f"""
        Human: You are an AI assistant specialized in Milvus Database Documentation.
        Use the context below to provide accurate, fact-based answers about Milvus.
        
        Previous conversation:
        {history_context}

        <context>
        {context}
        </context>

        <question>
        {question}
        </question>

        Assistant:"""
    
    response = llm.invoke(prompt)
    return response, len(search_res[0])

def direct_response(llm, question, chat_history):
    """Handle questions without retrieval"""
    history_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history[-3:]])
    
    prompt = f"""
        Human: You are an AI assistant for a Milvus documentation system.
        Answer the question directly without needing additional context.
        
        Previous conversation:
        {history_context}
        
        Question: {question}
        
        Assistant:"""
    
    response = llm.invoke(prompt)
    return response

def optimized_rag_query(client, llm, collection_name, question, chat_history):
    """Two-stage approach: check if retrieval needed"""
    if not needs_retrieval(llm, question):
        return direct_response(llm, question, chat_history), 0
    
    return rag_query_with_retrieval(client, llm, collection_name, question, chat_history)