# app/vision_helper.py

import sys
import ollama
import base64

def main():
    """
    A simple, dedicated script to get a multimodal response from Ollama.
    This script is intended to be called from another process (like our main app).
    """
    if len(sys.argv) != 3:
        print("Error: This script requires two arguments: an image path and a prompt.", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]
    prompt_text = sys.argv[2]

    try:
        # Prepare the image in the format Ollama needs
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Connect to the Ollama client (this happens inside WSL)
        client = ollama.Client()

        # Send the request with both image and text
        response = client.chat(
            model='hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL',
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [encoded_image]
                }
            ]
        )
        
        # Print the successful response to standard output
        print(response['message']['content'])

    except Exception as e:
        # Print any errors to standard error
        print(f"Error in vision_helper.py: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()