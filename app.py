import flask 
import io
from PIL import Image
import numpy as np
from flask_cors import CORS
import face_recognition
## import cv2 
import json

app = flask.Flask(__name__)
CORS(app)

class DefaultConfig:
    img_size: int = 1000
    labels: list = ['cat', 'dog']

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    # image = image.resize(target)
    image = np.asarray(image)
    return image

@app.route("/embeddings", methods=["POST"])
def predict():
    data = {}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            data["old"] = [image.size[0], image.size[1]]
            image = prepare_image(image, 
            target=(DefaultConfig.img_size,DefaultConfig.img_size)
            )
            try:
                bounding_boxes = face_recognition.face_locations(image)
                landmarks = face_recognition.face_landmarks(image)
                latent_features = face_recognition.face_encodings(image)[0]
                data["bounding_boxes"] = json.dumps(bounding_boxes)
                data["landmarks"] = json.dumps(landmarks)
                data["latent_features"] = json.dumps(latent_features.tolist())
            except:
                data["bounding_boxes"] = json.dumps([])
                data["landmarks"] = json.dumps([])
                data["latent_features"] = json.dumps([])
            data["success"] = True
    return flask.jsonify(data)

if __name__ == "__main__":
    app.run()