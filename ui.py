<<<<<<< HEAD
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import os


# Set up the model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    return processor, model, device


def generate_captions(processor, model, device, image, context_text=""):
    # Prepare inputs
    inputs = processor(images=image, text=context_text, return_tensors="pt").to(
        device, torch.float16
    )

    # Generate concise caption
    concise_output = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        length_penalty=0.8,  # Prefer shorter captions
        temperature=0.7,
    )
    concise_caption = processor.decode(concise_output[0], skip_special_tokens=True)

    # Generate detailed caption
    detailed_output = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        length_penalty=1.5,  # Encourage longer descriptions
        temperature=0.9,
        do_sample=True,
    )
    detailed_caption = processor.decode(detailed_output[0], skip_special_tokens=True)

    # Simple confidence estimation (placeholder - would need proper calibration)
    concise_confidence = min(0.95, max(0.7, 1 - (len(concise_caption.split()) / 100)))
    detailed_confidence = min(0.90, max(0.6, 1 - (len(detailed_caption.split()) / 200)))

    return concise_caption, detailed_caption, concise_confidence, detailed_confidence


def overlay_text_on_image(
    image, concise_text, detailed_text, concise_conf, detailed_conf
):
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Calculate text positions
    width, height = image.size
    margin = 10

    # Draw concise caption (blue) at top
    concise_position = (margin, margin)
    concise_text_with_conf = f"{concise_text} (Confidence: {concise_conf:.2f})"
    draw.text(concise_position, concise_text_with_conf, fill="blue", font=font)

    # Draw detailed caption (red) at bottom
    detailed_lines = []
    words = detailed_text.split()
    current_line = []
    max_chars_per_line = width // 10  # Rough estimate

    for word in words:
        if len(" ".join(current_line + [word])) <= max_chars_per_line:
            current_line.append(word)
        else:
            detailed_lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        detailed_lines.append(" ".join(current_line))

    detailed_text_with_conf = (
        "\n".join(detailed_lines) + f"\n(Confidence: {detailed_conf:.2f})"
    )

    detailed_position = (margin, height - (len(detailed_lines) + 1) * 25 - margin)
    draw.multiline_text(
        detailed_position, detailed_text_with_conf, fill="red", font=font, spacing=4
    )

    return image


def main():
    st.title("Context-Aware Image Caption Generator")
    st.write(
        "Upload an image and provide contextual information for enhanced captioning"
    )

    # Load model
    processor, model, device = load_model()

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Context inputs
    st.subheader("Contextual Information (Optional)")
    section_header = st.text_input("Section Header")
    above_text = st.text_area("Text Above Image")
    below_text = st.text_area("Text Below Image")
    footnote = st.text_input("Footnote")
    existing_caption = st.text_input("Existing Caption (if any)")

    # Combine context
    context_parts = []
    if section_header:
        context_parts.append(f"Section Header: {section_header}")
    if above_text:
        context_parts.append(f"Text Above Image: {above_text}")
    if below_text:
        context_parts.append(f"Text Below Image: {below_text}")
    if footnote:
        context_parts.append(f"Footnote: {footnote}")
    if existing_caption:
        context_parts.append(f"Existing Caption: {existing_caption}")

    context_text = " ".join(context_parts)

    if uploaded_file is not None and st.button("Generate Captions"):
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Generate captions
        with st.spinner("Generating captions..."):
            concise_cap, detailed_cap, concise_conf, detailed_conf = generate_captions(
                processor, model, device, image, context_text
            )

        # Display text results
        st.subheader("Generated Captions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Concise Caption (Confidence: {concise_conf:.2f})**")
            st.info(concise_cap)
        with col2:
            st.markdown(f"**Detailed Caption (Confidence: {detailed_conf:.2f})**")
            st.error(detailed_cap)

        # Create and display overlaid image
        overlaid_image = overlay_text_on_image(
            image.copy(), concise_cap, detailed_cap, concise_conf, detailed_conf
        )

        st.subheader("Image with Overlaid Captions")
        st.image(overlaid_image, use_column_width=True)

        # Download button
        st.download_button(
            label="Download Captioned Image",
            data=overlaid_image.tobytes(),
            file_name="captioned_" + uploaded_file.name,
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
=======
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import os


# Set up the model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    return processor, model, device


def generate_captions(processor, model, device, image, context_text=""):
    # Prepare inputs
    inputs = processor(images=image, text=context_text, return_tensors="pt").to(
        device, torch.float16
    )

    # Generate concise caption
    concise_output = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=5,
        length_penalty=0.8,  # Prefer shorter captions
        temperature=0.7,
    )
    concise_caption = processor.decode(concise_output[0], skip_special_tokens=True)

    # Generate detailed caption
    detailed_output = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        length_penalty=1.5,  # Encourage longer descriptions
        temperature=0.9,
        do_sample=True,
    )
    detailed_caption = processor.decode(detailed_output[0], skip_special_tokens=True)

    # Simple confidence estimation (placeholder - would need proper calibration)
    concise_confidence = min(0.95, max(0.7, 1 - (len(concise_caption.split()) / 100)))
    detailed_confidence = min(0.90, max(0.6, 1 - (len(detailed_caption.split()) / 200)))

    return concise_caption, detailed_caption, concise_confidence, detailed_confidence


def overlay_text_on_image(
    image, concise_text, detailed_text, concise_conf, detailed_conf
):
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Calculate text positions
    width, height = image.size
    margin = 10

    # Draw concise caption (blue) at top
    concise_position = (margin, margin)
    concise_text_with_conf = f"{concise_text} (Confidence: {concise_conf:.2f})"
    draw.text(concise_position, concise_text_with_conf, fill="blue", font=font)

    # Draw detailed caption (red) at bottom
    detailed_lines = []
    words = detailed_text.split()
    current_line = []
    max_chars_per_line = width // 10  # Rough estimate

    for word in words:
        if len(" ".join(current_line + [word])) <= max_chars_per_line:
            current_line.append(word)
        else:
            detailed_lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        detailed_lines.append(" ".join(current_line))

    detailed_text_with_conf = (
        "\n".join(detailed_lines) + f"\n(Confidence: {detailed_conf:.2f})"
    )

    detailed_position = (margin, height - (len(detailed_lines) + 1) * 25 - margin)
    draw.multiline_text(
        detailed_position, detailed_text_with_conf, fill="red", font=font, spacing=4
    )

    return image


def main():
    st.title("Context-Aware Image Caption Generator")
    st.write(
        "Upload an image and provide contextual information for enhanced captioning"
    )

    # Load model
    processor, model, device = load_model()

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Context inputs
    st.subheader("Contextual Information (Optional)")
    section_header = st.text_input("Section Header")
    above_text = st.text_area("Text Above Image")
    below_text = st.text_area("Text Below Image")
    footnote = st.text_input("Footnote")
    existing_caption = st.text_input("Existing Caption (if any)")

    # Combine context
    context_parts = []
    if section_header:
        context_parts.append(f"Section Header: {section_header}")
    if above_text:
        context_parts.append(f"Text Above Image: {above_text}")
    if below_text:
        context_parts.append(f"Text Below Image: {below_text}")
    if footnote:
        context_parts.append(f"Footnote: {footnote}")
    if existing_caption:
        context_parts.append(f"Existing Caption: {existing_caption}")

    context_text = " ".join(context_parts)

    if uploaded_file is not None and st.button("Generate Captions"):
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Display original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Generate captions
        with st.spinner("Generating captions..."):
            concise_cap, detailed_cap, concise_conf, detailed_conf = generate_captions(
                processor, model, device, image, context_text
            )

        # Display text results
        st.subheader("Generated Captions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Concise Caption (Confidence: {concise_conf:.2f})**")
            st.info(concise_cap)
        with col2:
            st.markdown(f"**Detailed Caption (Confidence: {detailed_conf:.2f})**")
            st.error(detailed_cap)

        # Create and display overlaid image
        overlaid_image = overlay_text_on_image(
            image.copy(), concise_cap, detailed_cap, concise_conf, detailed_conf
        )

        st.subheader("Image with Overlaid Captions")
        st.image(overlaid_image, use_column_width=True)

        # Download button
        st.download_button(
            label="Download Captioned Image",
            data=overlaid_image.tobytes(),
            file_name="captioned_" + uploaded_file.name,
            mime="image/jpeg",
        )


if __name__ == "__main__":
    main()
>>>>>>> master
