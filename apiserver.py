import json
from flask import Flask, request, jsonify
import numpy as np
from mordecai import Geoparser

app = Flask(__name__)
geo = Geoparser()

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

@app.route('/geoparse', methods=['POST'])
def geoparse():
    data = request.get_json()
    text = data['text']

    result = geo.geoparse(text)
    result_serializable = json.loads(json.dumps(result, default=convert_to_serializable))

    return jsonify(result=result_serializable)

if __name__ == '__main__':
    app.run()
