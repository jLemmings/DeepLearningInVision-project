import json
import os

import cv2
import numpy
import pytesseract
from PIL import Image
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# a = parse(image_path, False, False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/readimg", methods=['POST'])
def readimg():
    # read image file string data
    filestr = request.files['file'].read()
    # convert string data to numpy array
    npimg = numpy.frombuffer(filestr, numpy.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)  # Create a temp file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)  # Remove the temp file
    return jsonify(text)

@app.route("/readstr", methods=['POST'])
def readstr():
    # Do magic
    #print("REQUEST: ", request.get_json(force=True))
    string_to_summarize = request.get_json()
    print("STRING: ", string_to_summarize)
    return jsonify("Magic summary")


if __name__ == '__main__':
    app.run()
