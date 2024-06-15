import os
import openai
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

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

def main():
    print("Welcome to the CLI-based chatbot. Before we begin, I would like to know your name and occupation/company.")
    
    visitor_name = input("Please enter your name: ")
    visitor_occupation = input("Please enter your occupation/company: ")

    with open("visitor_info.txt", "a") as file:
        file.write(f"Name: {visitor_name}\nOccupation/Company: {visitor_occupation}\n\n")

    print(f"Thank you, {visitor_name}. You can now ask me any questions you want to know about me.")
    
    conversation_history = []
    
    while True:
        message = input(f"{visitor_name}: ")
        if message.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        print("Chatbot: Typing...")

        result = generate_response(message, conversation_history, visitor_name)
        conversation_history.append(f"{visitor_name}: {message}")
        conversation_history.append(f"Chatbot: {result}")

        print(f"Chatbot: {result}")

if __name__ == '__main__':
    main()
