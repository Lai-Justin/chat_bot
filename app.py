from flask import Flask, request, jsonify, render_template
from model import NeuralNetwork
import torch
import json
import random
from nltk_utils import tokenize, bag_of_words

app = Flask(__name__)

# Load the trained model and other necessary data
# Make sure to replace these placeholders with your actual model loading code
with open('intents.json', 'r') as f:
    intents = json.load(f)



FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=output_size)
model.load_state_dict(model_state)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    
        
    
    # Process user message
    tokens = tokenize(message)
    
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X_tensor = torch.from_numpy(X).float()
    # Pass user message to the chatbot model
    # Make sure to replace this placeholder with your actual model inference code
    output = model(X_tensor)
    _, predicted = torch.max(output, dim = 1)
    tag = tags[predicted.item()]
    
    # Dummy response (replace with actual chatbot response)
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = f"Chatbot: {random.choice(intent['responses'])}"

    else:
        response = "Chatbot: I do not understand"
    
    return jsonify({'user_message': message, 'chatbot_response': response})

if __name__ == '__main__':
    app.run(debug=True)
