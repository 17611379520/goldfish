import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load pre-trained ResNet50 model
model = ResNet50(weights='C:/Users/fky/Desktop/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

# Function to process and classify uploaded image
def classify_image(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Perform the prediction
    predictions = model.predict(image)
    labels = decode_predictions(predictions, top=5)[0]

    # Display the results
    print(f"\nResults for image '{image_path}':")
    for label in labels:
        print(f"{label[1]}: {label[2]*100:.2f}%")

# Specify the image file path
image_path = 'C:/Users/fky/Desktop/ffish/C.jpg'

# Perform image classification and display results
classify_image(image_path)