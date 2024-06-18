from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Step 3: Define Constants
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Step 4: Define Helper Functions
def set_custom_prompt():
    """
    Create a custom prompt template for QA retrieval.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    """
    Create a RetrievalQA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

def load_llm():
    """
    Load the LLM (Language Learning Model).
    """
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=512, temperature=0.5)
    return llm

def qa_bot():
    """
    Initialize the QA bot with retrieval QA chain.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    """
    Get the final result of the bot for the given query.
    """
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Step 5: Run the Chatbot
print("Bot: Hi, I'm your Medical Bot. How can I assist you today?")
print("Bot: You can type 'quit' to exit the chat.")

while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        print("Bot: Goodbye!")
        break
    response = final_result(user_input)
    print("Bot:", response)
