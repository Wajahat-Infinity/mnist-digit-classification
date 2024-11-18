import requests
from PIL import Image
import io

# Open an image
image = Image.open('2.png')  # Replace with your image file
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

# Send the image to the Flask API
response = requests.post('http://127.0.0.1:5000/predict', files={'file': img_byte_arr})

# Print the predicted digit
print(response.json())  # Expected output: {'digit': <predicted digit>}
