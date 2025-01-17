import streamlit as st
from langchain_openai import ChatOpenAI
import os
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

# Set page config
st.set_page_config(page_title="Icicibank Assistant", layout="wide")

# Streamlit app header
st.title("Icicibank Customer Support Chatbot")

# Sidebar for API Key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Main app logic
if "OPENAI_API_KEY" in os.environ:
    # Initialize components
    @st.cache_resource
    def initialize_components():
        dotenv.load_dotenv()
        chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

        loader = WebBaseLoader("https://www.icicibank.com/about-us")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(k=4)

        SYSTEM_TEMPLATE = """
        You are a helpful assistant chatbot for Icicibank. Your knowledge comes exclusively from the content of our website. Please follow these guidelines:
        1. When user Greets start by greeting the user warmly. For example: "Hello! Welcome to Icicibank. How can I assist you today?"
        2. When answering questions, use only the information provided in the website content. Do not make up or infer information that isn't explicitly stated.
        3. If a user asks a question that can be answered using the website content, provide a clear and concise response. Include relevant details, but try to keep answers brief and to the point.
        4. If a user asks a question that cannot be answered using the website content, or if the question is unrelated to Icicibank, respond politely with something like:
           "I apologize, but I don't have information about that topic. My knowledge is limited to Icicibank's products/services and the content on our website. Is there anything specific about Icicibank I can help you with?"
        5. Always maintain a friendly and professional tone.
        6. If you're unsure about an answer, it's okay to say so. You can respond with:
           "I'm not entirely sure about that. To get the most accurate information, I'd recommend checking our website or contacting our customer support team."
        7. If a user asks for personal opinions or subjective information, remind them that you're an AI assistant and can only provide factual information from the website.
        8. End each interaction by asking if there's anything else you can help with related to Icicibank.
        Remember, your primary goal is to assist users with information directly related to Icicibank and its website content. Stick to this information and avoid speculating or providing information from other sources.
        If the user query is not in context, simply tell We are sorry, we don't have information on this        <context>
        {context}
        </context>
        Chat History:
        {chat_history}
        """

        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

        return retriever, document_chain, memory

    # Load components
    with st.spinner("Initializing Icicibank Assistant..."):
        retriever, document_chain, memory = initialize_components()

    # Chat interface
    st.subheader("Chat with Icicibank Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know about Icicibank?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

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

            # The response is already a string, so we can use it directly
            full_response = response
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Update memory
        memory.save_context({"input": prompt}, {"output": full_response})

else:
    st.warning("Please enter your OpenAI API Key in the sidebar to start the chatbot.")

# Add a footer
st.markdown("---")
st.markdown("By Swarno")