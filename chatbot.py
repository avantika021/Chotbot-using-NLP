import nltk
import random
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Download necessary NLTK packages (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')

# Define intents and responses (For demonstration purposes)
intents = {
    'greeting': ['Hello', 'Hi', 'Hey', 'Greetings'],
    'goodbye': ['Goodbye', 'See you later', 'Bye', 'Take care'],
    'weather': ['What is the weather like?', 'Tell me the weather', 'How is the weather today?'],
    'jokes': ['Tell me a joke', 'Make me laugh', 'Tell me something funny'],
}

# Define sample responses
responses = {
    'greeting': ['Hello! How can I help you today?', 'Hi! What can I do for you today?'],
    'goodbye': ['Goodbye! Have a great day!', 'See you later!'],
    'weather': ['The weather is sunny and 25°C.', 'It looks cloudy with a chance of rain today.'],
    'jokes': ['Why don’t skeletons fight each other? They don’t have the guts.', 
              'Why did the scarecrow win an award? Because he was outstanding in his field.'],
}

# Preprocess input text (tokenization, stop word removal, and lemmatization)
def preprocess_input(user_input):
    # Tokenize and convert to lowercase
    tokens = nltk.word_tokenize(user_input.lower())
    
    # Remove punctuation and stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)

# Match user input to intent and generate a response
def chatbot_response(user_input):
    user_input = preprocess_input(user_input)

    # Simple matching (this can be improved with machine learning models)
    for intent, patterns in intents.items():
        if any(pattern.lower() in user_input for pattern in patterns):
            return random.choice(responses[intent])

    return "Sorry, I didn't understand that. Can you please rephrase?"

