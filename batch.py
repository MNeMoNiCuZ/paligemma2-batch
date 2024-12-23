from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import os
import argparse
import glob
from colorama import init, Fore
from datetime import datetime
import re
from tqdm import tqdm
import psutil
import warnings

# Configuration Settings
overwrite_captions = True
max_batch_size = 4
output_format = ".txt"
prompt = "<image>Caption the image with a simple short caption"

# Output Settings
prepend_string = ""
append_string = ""
print_captions = False

# Generation Settings
model_name = "google/paligemma2-10b-ft-docci-448"
quantization_bits = 4  # Set to None for full precision, 4 for 4-bit, or 8 for 8-bit quantization
min_tokens = 20
max_tokens = 512
max_word_character_length = 30
repetition_penalty = 1.15

# Cleanup Settings
max_retries = 10
prune_end = True
retry_words = ["no_parallel"]
remove_words = ["#", "/", "、", "@", "__", "|", "  ", ";", "~", "\"", "*", "^", ",,", "ON DISPLAY:"]
strip_contents_inside = ["(", "[", "{"]
remove_underscore_tags = True

# Argument Parsing
parser = argparse.ArgumentParser(description="Process images and generate captions using PaliGemma.")
parser.add_argument('--input_folder', type=str, help='Input folder path')
parser.add_argument('--output_folder', type=str, help='Output folder path')
parser.add_argument('--batch_size', type=int, help='Batch size for processing')
parser.add_argument('--prompt', type=str, help='Prompt for image captioning')
parser.add_argument('--output_format', type=str, help='Output file format')
parser.add_argument('--prepend_string', type=str, help='Prepend string for captions')
parser.add_argument('--append_string', type=str, help='Append string for captions')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing captions')
parser.add_argument('--quantization_bits', type=int, help='Quantization bits for model')
parser.add_argument('--min_tokens', type=int, help='Minimum number of tokens for generation')
parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens for generation')
parser.add_argument('--model_name', type=str, help='Model name for image captioning')

args = parser.parse_args()

# Override defaults with arguments (if provided)
if args.input_folder:
    input_dir = args.input_folder
if args.output_folder:
    output_dir = args.output_folder
if args.prompt:
    prompt = args.prompt
if args.output_format:
    output_format = args.output_format
if args.batch_size:
    max_batch_size = args.batch_size
if args.prepend_string:
    prepend_string = args.prepend_string
if args.append_string:
    append_string = args.append_string
if args.overwrite:
    overwrite_captions = args.overwrite
if args.quantization_bits is not None:
    quantization_bits = args.quantization_bits
if args.min_tokens is not None:
    min_tokens = args.min_tokens
if args.max_tokens is not None:
    max_tokens = args.max_tokens
if args.model_name:
    model_name = args.model_name

# Path Settings
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, 'input')

# Check if the output_folder argument is provided
if args.output_folder:
    output_in_input_dir = False
    output_dir = args.output_folder  # Use the specified output folder
else:
    output_in_input_dir = True
    output_dir = os.path.join(script_dir, 'output')  # Default output folder if not specified

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Ensure output directory exists
if not args.input_folder:
    input_dir = os.path.join(script_dir, 'input')  # Default input folder if not specified

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.")

# Initialize colorama
init(autoreset=True)

# System info printing
def print_system_info():
    print(Fore.CYAN + "\nSystem Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f}GB")
    print()

# Load model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_system_info()
print(Fore.YELLOW + "Loading model and processor...")

try:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = PaliGemmaProcessor.from_pretrained(model_name)
    print(Fore.GREEN + "Model and processor loaded successfully.\n")
except Exception as e:
    print(Fore.RED + f"Error loading model or processor: {e}")
    raise

def prune_text(text):
    if not prune_end:
        return text
    last_period_index = text.rfind('.')
    last_comma_index = text.rfind(',')
    prune_index = max(last_period_index, last_comma_index)
    if prune_index != -1:
        return text[:prune_index].strip()
    return text

def contains_retry_word(text, retry_words):
    return any(word in text for word in retry_words)

def remove_unwanted_words(text, remove_words):
    for word in remove_words:
        text = text.replace(word, ' ')
    return text

def strip_contents(text, chars):
    for char in chars:
        if char == "(":
            text = re.sub(r'\([^)]*\)', ' ', text)
        elif char == "[":
            text = re.sub(r'\[[^\]]*\]', ' ', text)
        elif char == "{":
            text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s([,.!?;])', r'\1', text)
    text = re.sub(r'([,.!?;])\s', r'\1 ', text)
    return text.strip()

def remove_long_words(text, max_word_length):
    words = text.split()
    for i, word in enumerate(words):
        if len(word) > max_word_length:
            last_period_index = text.rfind('.', 0, text.find(word))
            last_comma_index = text.rfind(',', 0, text.find(word))
            prune_index = max(last_period_index, last_comma_index)
            if prune_index != -1:
                return text[:prune_index].strip()
            else:
                return text[:text.find(word)].strip()
    return text

def clean_text(text):
    text = remove_unwanted_words(text, remove_words)
    text = strip_contents(text, strip_contents_inside)
    text = remove_long_words(text, max_word_character_length)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if remove_underscore_tags:
        text = ' '.join([word for word in text.split() if '_' not in word])
    return text

def process_image_batch(batch, model, processor, device):
    batch_inputs = []
    batch_images = []
    batch_paths = []
    batch_output_paths = []

    # Prepare batch
    for image_path in batch:
        output_file_path = os.path.splitext(image_path)[0] + output_format if output_in_input_dir else \
            os.path.join(output_dir, os.path.splitext(os.path.relpath(image_path, input_dir))[0] + output_format)
        
        if os.path.exists(output_file_path) and not overwrite_captions:
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(device)
            
            batch_inputs.append(inputs)
            batch_images.append(image)
            batch_paths.append(image_path)
            batch_output_paths.append(output_file_path)
        except Exception as e:
            if print_captions:
                print(Fore.RED + f"Error preparing {image_path}: {e}")

    if not batch_inputs:
        return

    try:
        # Process batch
        input_lens = [inputs["input_ids"].shape[-1] for inputs in batch_inputs]
        concatenated_inputs = {
            k: torch.cat([inputs[k] for inputs in batch_inputs], dim=0)
            for k in batch_inputs[0].keys()
        }

        with torch.inference_mode():
            generations = model.generate(
                **concatenated_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=2,
                repetition_penalty=repetition_penalty
            )

        # Process outputs
        for i, (generation, input_len, image_path, output_path) in enumerate(zip(generations, input_lens, batch_paths, batch_output_paths)):
            generation = generation[input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            pruned_text = prune_text(decoded)
            cleaned_text = clean_text(pruned_text)
            final_text = f"{prepend_string}{cleaned_text}{append_string}"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_text)

            if print_captions:
                print(Fore.GREEN + f"Processed {image_path}")
                print(Fore.LIGHTBLACK_EX + f"Output: {final_text}\n")

    except Exception as e:
        if print_captions:
            print(Fore.RED + f"Error processing batch: {e}")

    # Cleanup
    del batch_inputs, concatenated_inputs
    torch.cuda.empty_cache()

# Process images
image_extensions = ['jpg', 'jpeg', 'png', 'webp']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_dir, '**', f'*.{ext}'), recursive=True))

total_files = len(image_paths)
existing_captions = sum(1 for path in image_paths if os.path.exists(os.path.splitext(path)[0] + output_format))
files_to_process = total_files if overwrite_captions else total_files - existing_captions
total_batches = (files_to_process + max_batch_size - 1) // max_batch_size

print(Fore.YELLOW + f"\nFile Statistics:")
print(f"Total image files found: {total_files}")
print(f"Existing caption files: {existing_captions}")
print(f"Files to be processed: {files_to_process}")
print(f"Batch size: {max_batch_size}")
print(f"Total batches: {total_batches}\n")

processed_count = 0
error_count = 0
start_time = datetime.now()

if files_to_process > 0:
    with tqdm(total=files_to_process, desc="Processing images", unit="img") as pbar:
        for i in range(0, len(image_paths), max_batch_size):
            batch = image_paths[i:i + max_batch_size]
            try:
                process_image_batch(batch, model, processor, device)
                processed_count += len(batch)
            except Exception as e:
                error_count += len(batch)
                if print_captions:
                    print(Fore.RED + f"Batch error: {e}")
            pbar.update(min(max_batch_size, files_to_process - i))

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(Fore.YELLOW + "\nProcessing Summary:")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average time per image: {duration/processed_count:.2f} seconds")
    print(f"Successfully processed: {processed_count} images")
    print(f"Failed to process: {error_count} images")
    print(f"Completion rate: {(processed_count/files_to_process)*100:.1f}%")
else:
    print(Fore.YELLOW + "No files to process.")