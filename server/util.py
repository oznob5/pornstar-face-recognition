"""
Base64 is the way to covert an image into a string.
It must done due to UI need to pass image to the backend
and backend will use the saved model to predict the class.
"""
import json
import cv2
import joblib
import numpy as np
import base64
from wavelet import w2d

# private variables
__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    images = get_cropped_image_if_two_eyes(file_path, image_base64_data)

    result = []

    for img in images:
        scalled_raw_img = cv2.resize(img, (32, 32))  # images could have different size, so i resize them
        img_haar = w2d(img, 'db1', 5)
        scalled_img_haar = cv2.resize(img_haar, (32, 32))
        # stack these images vertically
        combined_image = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_haar.reshape(32 * 32, 1)))

        final_image = combined_image.reshape(1, 4096).astype(float)  # 32*32*3 + 32*32

        result.append({
            'class' : class_number_to_name(__model.predict(final_image)[0]),
            'class_probability' : np.round(__model.predict_proba(final_image) * 100, 2).tolist()[0],
            'class_dictionary' : __class_name_to_number
        })

    return result

def load_saved_artifacts():
    print("loading saved artifacts start...")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dict.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v : k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts DONE.")

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def get_cv2_image_from_base64_string(b64str):
    """
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    taking base64 string and converting it to openCV image
    """
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_two_eyes(image_path, image_base64_data):
    """
    function detects two eyes on the image and cropp it, returns faces in array
    if there is no face in the picture, it will return empty array
    """
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x1, y1, x2, y2) in faces:
        roi_gray = gray[y1 : y1 + y2, x1 : x1 + x2]
        roi_color = img[y1 : y1 + y2, x1 : x1 + x2]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_eva():
    with open('b64.txt') as f:
        return f.read()

if __name__ == "__main__":
    pass
    # load_saved_artifacts()
    # print(classify_image(get_b64_test_image_for_eva()))
    # print(classify_image(None, './test_images/eva_elfie1.jpg'))
    # print(classify_image(None, './test_images/lana_rhoades2.jpg'))
    # print(classify_image(None, './test_images/nancy_ace1.jpg'))