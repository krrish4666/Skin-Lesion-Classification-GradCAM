import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
for layer in model.layers[::-1]:
    if 'conv' in layer.name:
        print(layer.name)
        break
last_conv_layer_name = "top_conv"
from tensorflow.keras.models import Model

grad_model = Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)
img = image.load_img("/content/skin_cancer_data/HAM10000_images_part_1/ISIC_0024307.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

print(img_array.shape)  # Should be (1, 224, 224, 3)
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
import cv2
import matplotlib.pyplot as plt

def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # ensure size matches heatmap

    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # resize to match image

    # Ensure both are uint8 and same shape
    superimposed_img = cv2.addWeighted(img.astype(np.uint8), 1-alpha, heatmap.astype(np.uint8), alpha, 0)

    # Show
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM")
    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.show()

import matplotlib.pyplot as plt
import cv2
import numpy as np

# Example class names (HAM10000 classes)
class_names = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanocytic nevi",
    "Vascular lesions",
    "Melanoma"
]

def get_img_array(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, target_size)
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_resized

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(jet, alpha, img, 1 - alpha, 0)
    return superimposed_img

# List of image paths
img_paths = [
    "/content/skin_cancer_data/HAM10000_images_part_1/ISIC_0024307.jpg",
    "/content/skin_cancer_data/HAM10000_images_part_1/ISIC_0024308.jpg",
    "/content/skin_cancer_data/HAM10000_images_part_1/ISIC_0024309.jpg"
]

# Plot grid
fig, axes = plt.subplots(len(img_paths), 2, figsize=(8, 4*len(img_paths)))

for i, img_path in enumerate(img_paths):
    img_array, orig_img = get_img_array(img_path)
    
    # Predict class
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    
    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    superimposed_img = display_gradcam(orig_img, heatmap)
    
    # Original image
    axes[i, 0].imshow(orig_img)
    axes[i, 0].axis('off')
    axes[i, 0].set_title("Original")
    
    # Grad-CAM image + predicted label
    axes[i, 1].imshow(superimposed_img)
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f"Grad-CAM\nPred: {pred_class}")

plt.tight_layout()
plt.show()
