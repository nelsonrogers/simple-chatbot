import streamlit as st
import sqlite3
from transformers import pipeline, Conversation, AutoTokenizer
from uuid import uuid4
from datetime import datetime

@st.cache_resource # load the model only once
def load_model():
    print("Loading model...")
    return pipeline(task="conversational", model="facebook/blenderbot-400M-distill")

chatbot = load_model()

@st.cache_resource # load the tokenizer only once
def load_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    chat = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    tokenizer.apply_chat_template(chat, tokenize=False)
    return tokenizer

tokenizer = load_tokenizer()

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('chat_history.db')
c = conn.cursor()

@st.cache_resource # create the table only once
def create_table():
    print("Creating SQLite table...")
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
        )
    ''')
    conn.commit()

create_table()

# Function to get chat history from the database
def get_chat_history():
    c.execute("SELECT role, content FROM chat_history ORDER BY timestamp ASC")
    return c.fetchall()

# Function to add a new message to the chat history
def add_to_chat_history(role, content):
    # add timestamp to the message
    c.execute("INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)", (role, content, datetime.now()))
    conn.commit()

st.title("Nelson's Simple Chatbot")

# Display chat history
chat_history = get_chat_history()
for role, content in chat_history:
    st.write(f"{role.capitalize()}: {content}")

# Input area for new messages
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You: ", "")
    submit_button = st.form_submit_button("Submit")

# Option to start a new chat
if st.button("Start New Chat"):
    c.execute("DELETE FROM chat_history")
    conn.commit()
    st.rerun()

# Process user input
if submit_button and user_input:

    add_to_chat_history('user', user_input)

    full_chat = [{"role": role, "content": content} for role, content in get_chat_history()]

    # Process full chat history into single string to count tokens
    chat_input = " ".join([f"{message['role']}: {message['content']}" for message in full_chat])

    # Check if the token limit is exceeded
    if len(tokenizer.tokenize(chat_input)) < 128:
        
        conversation = Conversation(messages=full_chat)
        conversation = chatbot(conversation)
        
        response = conversation[-1]['content'].strip()
        add_to_chat_history('assistant', response)

        # Refresh the app to display the new chat
        st.rerun()
    
    else:
        st.write("The token limit is exceeded. Please start a new chat.")
