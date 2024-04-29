import streamlit as st
import requests
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# Load Mask2Former model and processor
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")

def perform_segmentation(image):
    # Prepare input image
    inputs = processor(images=image, return_tensors="pt")

    # Perform segmentation
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process segmentation results
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def main():
    st.title("Sidewalk Segmentation App")

    # Upload image or provide URL
    option = st.radio("Choose input option:", ("Upload Image", "Provide Image URL"))
    
    if option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
    else:
        image_url = st.text_input("Enter image URL:")
        if image_url:
            image = Image.open(requests.get(image_url, stream=True).raw)

    # Perform segmentation if image is provided
    if 'image' in locals():
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform sidewalk segmentation
        segmentation_result = perform_segmentation(image)

        # Convert segmentation result tensor to NumPy array
        segmentation_array = segmentation_result.cpu().numpy()

        # Visualize the segmentation result
        st.image(segmentation_array, caption="Segmentation Result (Raw)", use_column_width=True)

        # Threshold the segmentation result to get the sidewalk mask
        threshold = 0.5  # Adjust this threshold as needed
        sidewalk_mask = (segmentation_array > threshold).astype(np.uint8)

        # Calculate the confidence score
        score = np.mean(segmentation_array)

        # Determine the label based on whether the sidewalk mask contains any non-zero values
        if np.any(sidewalk_mask):
            label = "Sidewalk"
        else:
            label = "No Sidewalk"

        # Create PIL Image object from the sidewalk mask
        mask_image = Image.fromarray(sidewalk_mask * 255)

        # Display the segmentation result, score, and label
        st.image(mask_image, caption=f"Segmentation Result (Score: {score:.4f}, Label: {label})", use_column_width=True)
    else:
        st.warning("Please upload an image or provide an image URL.")

if __name__ == "__main__":
    main()
