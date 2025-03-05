from main import ChatBot
import streamlit as st

# Create an instance of the ChatBot
bot = ChatBot()

# Set Streamlit page configuration
st.set_page_config(page_title="Resume Analysis Bot", page_icon=":guardsman:", layout="wide")

# Sidebar Title
with st.sidebar:
    st.title('Resume Analysis Bot')

# Function for generating LLM response
def generate_response(input):
    return bot.generate_response(input)

# Store LLM generated responses in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me to analyze a resume."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input for questions
if user_input := st.chat_input("Ask about the resume..."):
    # Append user input to messages
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response only if the last message was from the user
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Analyzing the resume..."):
                response = generate_response(user_input)  # Get response from the bot
                st.write(response)  # Show the response
        # Store the assistant's response in the message history
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
