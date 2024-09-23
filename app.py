from flask import Flask, request, jsonify
import base64
from PIL import Image, ExifTags
import face_recognition
import io

app = Flask(__name__)

# Define the desired size
DESIRED_SIZE = (127, 169)  # (width, height)

@app.route('/upload', methods=['POST'])
def upload_image():
    image_data = request.json.get('image')

    if not image_data:
        return jsonify({"message": "No image data provided."}), 400
    # Decode the base64 image
    img_data = base64.b64decode(image_data.split(',')[1])  # Remove metadata
    # Open the image and adjust for orientation
    image = Image.open(io.BytesIO(img_data))

    # Adjust for orientation
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()  # Get EXIF data
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"Error adjusting orientation: {e}")

    # Resize the image
    resized_image = image.resize(DESIRED_SIZE)

    # Save the resized image temporarily
    resized_image.save("received_image.jpg")

    # Load the reference image
    img1 = face_recognition.load_image_file('20231013_012013.jpg')
    img2 = face_recognition.load_image_file('received_image.jpg')

    # Extract the face encodings from the images
    face_encodings1 = face_recognition.face_encodings(img1)
    face_encodings2 = face_recognition.face_encodings(img2)

    # Check if faces were found in both images
    if not face_encodings1:
        return jsonify({"message": "No face detected in the reference image."}), 400
    if not face_encodings2:
        return jsonify({"message": "No face detected in the uploaded image."}), 400

    # Compare the face encodings
    result = face_recognition.compare_faces([face_encodings1[0]], face_encodings2[0])

    # Check if the faces match
    if result[0]:
        return jsonify({"message": "The images are of the same person."}), 200
    else:
        return jsonify({"message": "The images are of different persons."}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
