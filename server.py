from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import dotenv

app = Flask(__name__)

CORS(app)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/chat")
def search():
    prompt = request.args.get("prompt")

    print(prompt)
    answer = chat_with_icicibank_assistant(prompt)
    return jsonify({"response": answer})


# Load OpenAI API Key from environment
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Chat Model
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

# WebLoader setup
loader = WebBaseLoader("https://www.icicibank.com/about-us")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=4)

# System prompt templates
SYSTEM_TEMPLATE = """
You are a helpful assistant chatbot for Icicibank. Your knowledge comes exclusively from the content of our website. Please follow these guidelines:
1. When user Greets start by greeting the user warmly...
2. When answering questions, use only the information provided in the website content...
...
If the user query is not in context, simply tell We are sorry, we don't have information on this        <context>
{context}
Chat History:
{chat_history}
"""

# Create the chat prompt
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Memory setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)


# Function to process input and get response
def chat_with_icicibank_assistant(prompt):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(prompt)

    # Generate response
    response = document_chain.invoke(
        {
            "context": docs,
            "chat_history": memory.load_memory_variables({})["chat_history"],
            "messages": [
                HumanMessage(content=prompt)
            ],
        }
    )

    # Return the response
    full_response = response
    return full_response


# Example usage:
if __name__ == "__main__":
    app.run(debug=True)
