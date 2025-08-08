import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv
load_dotenv()

collection_name = os.getenv("OLLAMA_COLLECTION_NAME") or "demo_collection"
question = "How is data stored in milvus?"
llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b")
llm = OllamaLLM(model=llm_model )

embeding_model =  os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")
embeddings = OllamaEmbeddings(model=embeding_model)

docs = [] # add docs from loader

vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name=collection_name,
    drop_old=True,  # Drop the old Milvus collection if it exists
)

query = question
vectorstore.similarity_search(query, k=1)

# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()

# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""
 

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()



# Define a function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain.get_graph().print_ascii()

# Invoke the RAG chain with a specific question and retrieve the response
res = rag_chain.invoke({"question": question})
print(res)
