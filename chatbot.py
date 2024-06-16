import os
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Load environment variables
load_dotenv()

# Set OpenAI API key
#print(os.getenv("OPENAI_API_KEY"))
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Load data from CSV
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

# Create embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Use GPT-3.5 instead of GPT-4
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))


# Define the template
template = """
You are Vishad Mehta, and you will be answering questions based on your personal experience. Before responding, please review the given information to understand my background better.

I will provide a question from a potential employer:

{question}

Here is the relevant information to help you with your response:

{relevant_data}

Remember to respond as Vishad Mehta, maintaining a courteous and professional tone.

Address the visitor by their name, {visitor_name}.

Limit your response to 200 words or less, ensuring it is pertinent to the question.

Avoid phrases like "As Vishad Mehta, I would..." or "As Vishad Mehta, I will..." (this is already implied).

Do not include closing phrases such as "Warm regards", "Best regards", or any kind of signature at the end of your response.

For very personal questions, you may choose to answer with a witty response, keeping it concise and ideally within 50 words (use appropriate humor).

If a personal question is asked and relevant data is absent, you may choose to respond with a witty, brief reply, ideally within 50 words (use suitable jokes).

Do not provide any information that is not related to the question.

Formulate your answer as Vishad Mehta, making sure to include the data to address the potential employer's inquiry. Your response should be between 150-200 words, focusing on relevance to the question.

For questions related to your profession or skills, rely only on the provided data. For other questions where relevant data is missing, feel free to give a short, witty response in the first person, keeping it concise and ideally within 50 words.

Here is the previous conversation for context:
{conversation_history}
"""

prompt = PromptTemplate(
    input_variables=["question", "relevant_data", "visitor_name", "conversation_history"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(question, conversation_history, visitor_name):
    # Check if the input is a simple greeting
    greetings = ["hi", "hello", "hey"]
    if question.lower() in greetings:
        return f"Hello {visitor_name}, how can I assist you today?"
    
    relevant_data = retrieve_info(question)
    if not relevant_data:
        relevant_data = ["No relevant data found."]
    relevant_data = "\n".join(relevant_data)
    response = chain.run(
        question=question,
        relevant_data=relevant_data,
        visitor_name=visitor_name,
        conversation_history="\n".join(conversation_history)
    )
    return response

# FastAPI app
app = FastAPI()

#CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    visitor_name: str
    visitor_company: str
    conversation_history: List[str]

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = generate_response(request.question, request.conversation_history, request.visitor_name)
        
        # Save visitor name and company to visitor_info.txt
        visitor_info = f"Visitor Name: {request.visitor_name}\nCompany: {request.visitor_company}\n\n"
        with open('visitor_info.txt', 'a') as f:
            f.write(visitor_info)
        
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
