import json
import numpy as np
import time
import os
import json

def readJson(filename):
    data = json.load(open(filename))
    return data

def writeJson(filepath, data):
    json_str = json.dumps(data, indent=4)
    with open(filepath, 'w') as file:
        file.write(json_str)
