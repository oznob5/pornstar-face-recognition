from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/classify_image', methods = ['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']

    response = jsonify(util.classify_image(image_data))
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server for Pornstar Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)