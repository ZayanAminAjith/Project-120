#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# words to be igonred/omitted while framing the dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bow = []

    # tokenize the user_input
    token_1 = nltk.word_tokenize(user_input)
    token_2 = get_stem_words(token_1,ignore_words)
    token_2 = sorted(list(set(token_2)))
    for i in words:
        if i in token_2:
            bow.append(1)
        else:
            bow.append(0)
    bag.append(bow)
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # extract the class from the predicted_class_label
   predicted_class = classes[predicted_class_label]
   # now we have the predicted tag, select a random response
   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # choose a random bot response
        bot_response = random.choice(intent["responses"])
    
        return bot_response
    

print("Hi I am Stella, How Can I help you?")

while True:
    user_input = input("Type your message--")
    print("Users Input : ", user_input)
    bot = bot_response(user_input)
    print("Bots Input : ", bot)
    if user_input == "s":
        print("Sorry for you to leave")
        break