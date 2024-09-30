# Hindi & English OCR with Keyword Search

This project implements a web-based prototype for Optical Character Recognition (OCR) on images containing text in both Hindi and English. It also includes a basic keyword search functionality based on the extracted text.

## Features

- Upload and process images containing Hindi and English text
- Extract text from images using OCR
- Perform keyword search on the extracted text
- Web-based interface for easy interaction

## Technology Stack

- Python
- Hugging Face Transformers (Qwen2-VL-2B-Instruct model)
- PyTorch
- Gradio (for web interface)

## Setup and Installation

1. Clone the repository:
   ```
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open the provided URL in your web browser.

3. Upload an image containing Hindi and/or English text.

4. (Optional) Enter a keyword to search within the extracted text.

5. View the OCR results and any keyword matches.

## Limitations

- The current implementation uses CPU for processing, which may be slower for large images.

## Future Improvements

- Implement GPU support for faster processing
- Add support for multiple image uploads
- Enhance the user interface for better user experience

## Link

https://huggingface.co/spaces/pranshh/ocr-assignment

## Acknowledgements

This project uses the Qwen2-VL-2B-Instruct model from Hugging Face Transformers.