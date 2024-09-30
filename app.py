from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gradio as gr
from PIL import Image


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Initialize the model with float16 precision and handle fallback to CPU
# Simplified model loading function for CPU
def load_model():
    return Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True
    )

# Load the model
vlm = load_model()

# OCR function to extract text from an image
def ocr_image(image, query="Extract text from the image", keyword=""):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": query},
            ],
        }
    ]

    # Prepare inputs for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cpu")

    # Generate the output text using the model
    generated_ids = vlm.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    if keyword:
        keyword_lower = keyword.lower()
        if keyword_lower in output_text.lower():
            highlighted_text = output_text.replace(keyword, f"**{keyword}**")
            return f"Keyword '{keyword}' found in the text:\n\n{highlighted_text}"
        else:
            return f"Keyword '{keyword}' not found in the text:\n\n{output_text}"
    else:
        return output_text

# Gradio interface
def process_image(image, keyword=""):
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return ocr_image(image, keyword=keyword)

# Update the Gradio interface:
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Enter keyword to search (optional)")
    ],
    outputs="text",
    title="Hindi & English OCR with Keyword Search",
)

# Launch Gradio interface in Colab
interface.launch()