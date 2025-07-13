from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
import os

# 1. Login to Hugging Face Hub (required for gated models like Gemma)
# You can set your Hugging Face token as an environment variable (HF_TOKEN)
# or paste it directly here for demonstration purposes.
# For security, it's recommended to use an environment variable or a secrets manager.
# Example if using an environment variable: os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
# Or, if running in a Colab/Jupyter environment, you might use:
# from google.colab import userdata
# login(token=userdata.get('HF_TOKEN'))
# For this example, we'll assume you set it as an environment variable or
# will be prompted to log in if running in an interactive session.
try:
    login()
except Exception as e:
    print(f"Could not log in to Hugging Face Hub. Please ensure your token is set up correctly. Error: {e}")
    print("You might need to run `huggingface-cli login` in your terminal or set the HF_TOKEN environment variable.")
    exit()

#model_id = "google/gemma-3-12b-it"
#model_id = "google/gemma-3-4b-it"
model_id = "google/gemma-3-12b-it"
#it did not create snapshot folder with json files
# Define a directory to save the model (optional, defaults to Hugging Face cache)
#download_dir = "./models--google--gemma-3-4b-it"
download_dir = "./models--google--gemma-3-12b-it"

os.makedirs(download_dir, exist_ok=True)

print(f"Attempting to download model: {model_id} to {download_dir}")

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=download_dir)
    print("Tokenizer downloaded successfully.")

    # Load model
    # torch_dtype=torch.bfloat16 is often recommended for Gemma for performance and memory efficiency
    # device_map="auto" distributes the model across available devices (CPU/GPU)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=download_dir
    )
    print("Model downloaded successfully.")

    print(f"Model and tokenizer saved in: {download_dir}")

    # You can now use the model for inference, e.g.:
    # input_text = "What is the capital of France?"
    # input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)
    # outputs = model.generate(**input_ids, max_new_tokens=50)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print(f"An error occurred during model download: {e}")
    print("Please ensure you have accepted the Gemma model license on Hugging Face and your token is valid.")
    print("Also, check if you have sufficient disk space and GPU memory if device_map='auto' is used.")