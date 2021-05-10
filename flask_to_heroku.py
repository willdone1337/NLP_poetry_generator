import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin
from tensorflow.keras.models import load_model
import re
import os
# Declaring the flask app
app = Flask(__name__)

#Loading the model from pickle file
model = load_model('modelfornizami.h5')



# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():


    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

###poem_generator###
def poem(x):
    seed_text = ''
    for _ in range(1, next_words + 1):
        token_list = tokenizer.texts_to_sequences([seed_text + x])[0]
        token_list = pad_sequences([token_list], maxlen=max_lenght - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ''
        if _ % 4 == 0:
            seed_text = seed_text + ','
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        # if _%4==0:
        #   output_word+='\n'
        seed_text += ' ' + output_word

    split_regex = re.compile(r'[,]')
    sentences = [t.strip() for t in split_regex.split(seed_text)]
    poem = []
    for s in sentences:
        poem.append(s)
    return poem






# processing the request from dialogflow
def processRequest(req):
    result = req.get("queryResult")

    # Fetching the data points
    parameters = result.get("parameters")
    start_word = parameters.get("startword")
    str_features = start_word

    # Dumping the data into an array
    final_features = int_features

    # Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')

    # Fitting out model with the data points
    if (intent == 'Start_word'):
        prediction = poem(final_features)



        # Returning back the fullfilment text back to DialogFlow

        # log.write_log(sessionID, "Bot Says: "+fulfillmentText)
        return {
            "fulfillmentText": prediction
        }


if __name__ == '__main__':
    app.run()