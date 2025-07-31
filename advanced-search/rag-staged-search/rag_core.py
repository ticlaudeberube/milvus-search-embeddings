
from core import MilvusUtils

def needs_retrieval(llm, question, chat_history):
    """Stage 1: Check if question needs document retrieval"""
    recent_context = "\n".join([f"Q: {h['question']}" for h in chat_history[-2:]])
    
    classification_prompt = f"""
        Recent conversation:
        {recent_context}
        
        Current question: {question}
        
        Does this question ask about technical topics, database features, or need documentation?
        Consider: if recent conversation was about technical topics, follow-up questions likely need docs too.
        
        Answer ONLY "YES" or "NO".
        
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
    user_name = extract_user_name(chat_history)
    
    prompt = f"""
        Human: You are an AI assistant specialized in Milvus Database Documentation.
        Use the context below to provide accurate answers.
        {f"The user's name is {user_name}. Address them by name." if user_name else ""}
        
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

def extract_user_name(chat_history):
    """Extract user name from chat history"""
    for chat in reversed(chat_history):
        question_lower = chat['question'].lower()
        if "my name is" in question_lower:
            # Extract name after "my name is"
            parts = question_lower.split("my name is")
            if len(parts) > 1:
                name_part = parts[1].strip().split('.')[0].split(',')[0].split()[0]
                return name_part.capitalize()
        elif "i am" in question_lower:
            # Extract name after "i am"
            parts = question_lower.split("i am")
            if len(parts) > 1:
                name_part = parts[1].strip().split('.')[0].split(',')[0].split()[0]
                return name_part.capitalize()
    return ""

def direct_response(llm, question, chat_history):
    """Handle questions without retrieval"""
    history_context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in chat_history[-3:]])
    user_name = extract_user_name(chat_history)
    
    prompt = f"""
        Human: You are an AI assistant for a Milvus documentation system.
        {f"The user's name is {user_name}. Address them by name." if user_name else ""}
        
        Previous conversation:
        {history_context}
        
        Question: {question}
        
        Assistant:"""
    
    response = llm.invoke(prompt)
    return response

def optimized_rag_query(client, llm, collection_name, question, chat_history):
    """Two-stage approach: check if retrieval needed"""
    if not needs_retrieval(llm, question, chat_history):
        return direct_response(llm, question, chat_history), 0
    
    return rag_query_with_retrieval(client, llm, collection_name, question, chat_history)