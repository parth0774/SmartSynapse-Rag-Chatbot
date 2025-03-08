from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import json
import uuid
from datetime import datetime
import os
from hf_main import chatbot_service

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())

# Store chat history for each session
chat_sessions = {}

@app.route('/')
def index():
    # Initialize session if new user
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        chat_sessions[session['user_id']] = []
        
    return render_template('index.html', chat_history=chat_sessions.get(session['user_id'], []))

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get data from request
        user_message = request.form.get('msg')
        user_id = session.get('user_id', str(uuid.uuid4()))
        
        if not user_id in chat_sessions:
            chat_sessions[user_id] = []
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add user message to history
        chat_sessions[user_id].append({
            "role": "user",
            "content": user_message,
            "time": timestamp
        })
        
        # Get response from chatbot
        result = chatbot_service.get_response(user_message, chat_sessions[user_id])
        assistant_message = result["answer"]
        sources = result["sources"]
        
        # Add assistant message to history
        chat_sessions[user_id].append({
            "role": "assistant",
            "content": assistant_message,
            "time": timestamp,
            "sources": sources
        })
        
        # Prepare response
        response_data = {
            "message": assistant_message,
            "time": timestamp,
            "sources": sources
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "message": "Sorry, I encountered an error processing your request.",
            "time": datetime.now().strftime("%H:%M"),
            "sources": []
        })

@app.route('/clear', methods=['POST'])
def clear_chat():
    user_id = session.get('user_id')
    if user_id in chat_sessions:
        chat_sessions[user_id] = []
    return jsonify({"status": "success"})

@app.route('/add-source', methods=['POST'])
def add_source():
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({"status": "error", "message": "No URLs provided"})
        
        result = chatbot_service.add_documents(urls)
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Error adding source: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    feedback = data.get('feedback')
    message_id = data.get('messageId')
    
    # In a production app, store feedback in a database
    app.logger.info(f"Feedback received for message {message_id}: {feedback}")
    
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5003)