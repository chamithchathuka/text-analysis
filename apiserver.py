from flask import Flask, request, jsonify
from mordecai import Geoparser
import spacy
import json
import numpy as np

app = Flask(__name__)
geo = Geoparser()
nlp = spacy.load("en_core_web_sm")

def serialize_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))

def extract_attractions(text):
    doc = nlp(text)
    
    attractions = []
    
    for entity in doc.ents:
        if entity.label_ in ["FAC", "ORG"] and entity.label_ != "GPE":
            attractions.append(entity.text)
    
    return attractions

@app.route('/geoparse', methods=['POST'])
def geoparse():
    data = request.get_json()
    texts = data['texts']
    results = {}

    for text in texts:
        # Geoparse the text
        geoparse_result = geo.geoparse(text)

        # Text analysis
        doc = nlp(text.lower())

        implies_travel = False
        implies_stay = False
        implies_attraction = False
        attraction = None

        # Check for keywords related to air travel
        air_travel_keywords = ["flight", "fly", "airplane", "airport"]
        for token in doc:
            if token.text in air_travel_keywords:
                implies_travel = True
                break

        # Check for keywords related to staying at a hotel
        hotel_keywords = ["hotel", "accommodation", "check-in", "check-out"]
        for token in doc:
            if token.text in hotel_keywords:
                implies_stay = True
                break

        # Extract public attractions
        attractions = extract_attractions(text)
        if attractions:
            implies_attraction = True
            attraction = attractions[0]
                
        # Prepare the response
        analyzed_text = {
            'implies_travel': implies_travel,
            'implies_stay': implies_stay,
            'implies_attraction': implies_attraction,
            'attraction': attraction if implies_attraction else None
        }

        result = {
            'geoparse_result': json.loads(json.dumps(geoparse_result, default=serialize_numpy)),
            'text_analysis': {
                'input_text': text,
                'analyzed_text': analyzed_text
            }
        }

        results[text] = result

    return jsonify(results)

if __name__ == '__main__':
    app.run()
