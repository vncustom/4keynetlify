# -*- coding: utf-8 -*-
import os
import requests
import time
import re
from datetime import datetime
import traceback
from flask import Flask, request, jsonify
import json # Import json for potential parsing issues

# --- Configuration ---
# Read API keys from environment variables set in Netlify UI
# Name them KEY1, KEY2, KEY3, KEY4 in Netlify
API_KEYS = [
    os.environ.get('KEY1'),
    os.environ.get('KEY2'),
    os.environ.get('KEY3'),
    os.environ.get('KEY4'),
]
# Filter out any keys that weren't set
VALID_API_KEYS = [key for key in API_KEYS if key and key.strip()]

BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = 600 # 10 minutes timeout for API call (Netlify function timeout is shorter!)

# --- Text Splitting Logic (Adapted from original) ---

def smart_split_by_words(text, max_words):
    """Splits English text by word count, trying to keep sentences together."""
    if not text or not isinstance(text, str): return []
    # Use simpler sentence splitting for robustness
    sentences = re.split(r'([.!?])\s+', text.strip())
    combined_sentences = []
    temp_sentence = ""
    for part in sentences:
        if not part: continue
        temp_sentence += part
        if part in ".!?":
            combined_sentences.append(temp_sentence.strip())
            temp_sentence = ""
        else:
             temp_sentence += " " # Add space back if split removed it
    if temp_sentence.strip(): # Add last part if exists
        combined_sentences.append(temp_sentence.strip())

    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for sentence in combined_sentences:
        if not sentence.strip(): continue
        words_in_sentence = sentence.split()
        num_words = len(words_in_sentence)
        if num_words == 0: continue

        if num_words > max_words:
            # Force split long sentence
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words, current_word_count = [], 0
            for j in range(0, num_words, max_words):
                chunk_part = " ".join(words_in_sentence[j : j + max_words])
                chunks.append(chunk_part)
        elif current_word_count + num_words <= max_words:
            current_chunk_words.extend(words_in_sentence)
            current_word_count += num_words
        else:
            # Add current chunk, start new one
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            current_chunk_words = words_in_sentence
            current_word_count = num_words

    if current_chunk_words: # Add the last chunk
        chunks.append(" ".join(current_chunk_words))

    # Final filter for non-empty chunks
    return [chunk for chunk in chunks if chunk and chunk.strip()]


def split_text(text, split_method, language, split_length):
    """Splits text based on selected method. Returns list of parts or raises error."""
    parts = []
    if not text or not text.strip():
        raise ValueError("Text to process is empty.")

    try:
        # --- Split by Chapter ---
        if split_method == "Theo chương (第X章/Chương X)":
            # Regex: Allow optional leading whitespace, capture marker line, ensure it's a whole line.
            chapter_pattern = re.compile(
                 r'(?:^|\n)\s*(第\s*\d+\s*章.*?|Chương\s+\d+.*?)(?=\n|\Z)',
                 re.MULTILINE
            )
            # Find all markers (start index of the marker line)
            matches = list(chapter_pattern.finditer(text))
            markers = [match.start(1) for match in matches]

            if not markers:
                # Treat the whole text as one part if no markers found
                print("Warning: No chapter markers found. Processing text as a single part.")
                content = text.strip()
                if content: parts.append(content)
            else:
                num_markers = len(markers)
                for i in range(num_markers):
                    start_pos = markers[i]
                    end_pos = markers[i+1] if i + 1 < num_markers else len(text)
                    part_content = text[start_pos:end_pos].strip()
                    if part_content:
                        parts.append(part_content)

        # --- Split by Character/Word Count ---
        elif split_method == "Theo số ký tự/từ":
            if split_length <= 0:
                raise ValueError("Split length must be greater than 0.")

            if language == "ENG":
                parts = smart_split_by_words(text, split_length)
            else: # Character split (Vietnamese, Chinese, etc.)
                start = 0
                text_len = len(text)
                while start < text_len:
                    end = min(start + split_length, text_len)
                    break_point = end
                    # Try to break at sentence endings or newline within the desired range
                    if end < text_len:
                        found_break = -1
                        # Check common sentence endings and newline first, search backwards from 'end'
                        for bc in ".!?。\n":
                             # Search from start up to end (exclusive)
                             pos = text.rfind(bc, start, end)
                             # Choose the latest break point within the current segment
                             if pos > start and pos > found_break:
                                 found_break = pos

                        if found_break != -1:
                             break_point = found_break + 1 # Split after the punctuation/newline
                        else:
                             # If no sentence break, try space backwards
                             space_break = text.rfind(' ', start, end)
                             if space_break > start: # Ensure we don't get stuck
                                 break_point = space_break + 1 # Split after space

                    chunk = text[start:break_point].strip()
                    if chunk:
                        parts.append(chunk)
                    start = break_point # Move start to the break point for the next chunk

        else:
            raise ValueError("Invalid split method selected.")

        if not parts:
             # If splitting resulted in nothing (e.g., text was only whitespace), treat as single part if original text wasn't just whitespace
             content = text.strip()
             if content:
                  print("Warning: Splitting resulted in empty list, using original text as one part.")
                  parts.append(content)
             else:
                  raise ValueError("Text splitting resulted in no processable parts (input might be only whitespace or markers).")

        return parts

    except Exception as e:
        print(f"Error during text splitting: {e}")
        print(traceback.format_exc())
        # Re-raise a more specific error or return None/empty list? Re-raising is clearer.
        raise ValueError(f"Failed to split text: {e}")


# --- Flask App & Netlify Function Handler ---
app = Flask(__name__)

# This is the function Netlify will run.
# The route `/api/process` should match the fetch URL in index.html
# Using `app.route` makes it runnable locally (`python process.py`)
# Netlify typically picks up the `app` object.
@app.route('/api/process', methods=['POST'])
def handle_process():
    """Handles the POST request to process text."""
    if not VALID_API_KEYS:
        print("ERROR: No valid API keys found in environment variables (KEY1-KEY4).")
        return jsonify({"error": "Server configuration error: Missing API keys."}), 500

    num_available_keys = len(VALID_API_KEYS)
    print(f"Processing request with {num_available_keys} API key(s).")

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON body received."}), 400

        # --- Get Inputs ---
        language = data.get('language', '中文')
        model = data.get('model')
        split_method = data.get('split_method')
        split_length = data.get('split_length')
        prompt = data.get('prompt', '').strip()
        text_to_process = data.get('text_to_process', '').strip()

        # --- Input Validation ---
        if not model: return jsonify({"error": "Model not selected."}), 400
        if not split_method: return jsonify({"error": "Split method not selected."}), 400
        if not prompt: return jsonify({"error": "Prompt is required."}), 400
        if not text_to_process: return jsonify({"error": "Text to process is required."}), 400
        if split_method == "Theo số ký tự/từ" and (not isinstance(split_length, int) or split_length <= 0):
             return jsonify({"error": "Invalid split length for selected method."}), 400

        print(f"Received request: Model={model}, Lang={language}, Split={split_method}, Len={split_length if split_method == 'Theo số ký tự/từ' else 'N/A'}")

        # --- Split Text ---
        print("Splitting text...")
        parts = split_text(text_to_process, split_method, language, split_length)
        total_parts = len(parts)
        print(f"Text split into {total_parts} parts.")

        if total_parts == 0:
            return jsonify({"error": "Text could not be split into processable parts."}), 400

        # --- Process each part sequentially ---
        all_results = []
        errors_occurred = []
        start_time_total = time.time()

        for i, part_content in enumerate(parts, 1):
            part_start_time = time.time()
            print(f"--- Processing Part {i}/{total_parts} ---")

            # --- Select API Key ---
            key_index = (i - 1) % num_available_keys
            current_api_key = VALID_API_KEYS[key_index]
            key_log_name = f"Key #{key_index + 1}"
            print(f"Using {key_log_name}")

            # --- Prepare API Call ---
            full_prompt = f"{prompt}\n\n{part_content}"
            headers = {
                "Authorization": f"Bearer {current_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://YOUR_NETLIFY_SITE_NAME.netlify.app/", # Optional: Set your site URL
                "X-Title": "Netlify OpenRouter Client v1.0"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}]
                # Add other parameters like max_tokens if needed
                # "max_tokens": 8000,
            }

            # --- Make API Request ---
            try:
                response = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                part_api_time = time.time() - part_start_time
                print(f"Part {i} API call duration: {part_api_time:.2f}s, Status: {response.status_code}")

                # --- Handle API Response ---
                if response.status_code == 200:
                    try:
                        result_json = response.json()
                        choices = result_json.get('choices')
                        if (isinstance(choices, list) and choices and
                            isinstance(choices[0], dict) and 'message' in choices[0] and
                            isinstance(choices[0]['message'], dict) and 'content' in choices[0]['message']):

                            result_text = choices[0]['message']['content']
                            if result_text and isinstance(result_text, str) and result_text.strip():
                                all_results.append(f"## Part {i}/{total_parts} (Success)\n{result_text}\n")
                                print(f"Part {i} completed successfully.")
                            else:
                                err_msg = f"Part {i} API success (200) but response content is empty or invalid."
                                print(f"ERROR: {err_msg}")
                                errors_occurred.append(err_msg)
                                all_results.append(f"## Part {i}/{total_parts} (ERROR: Empty Content)\n{err_msg}\n")
                        else:
                             err_msg = f"Part {i} API success (200) but response JSON structure is invalid."
                             response_snippet = str(result_json)[:500]
                             print(f"ERROR: {err_msg} Response Snippet: {response_snippet}")
                             errors_occurred.append(f"{err_msg} - Snippet: {response_snippet}")
                             all_results.append(f"## Part {i}/{total_parts} (ERROR: Invalid JSON Structure)\n{err_msg}\nResponse Snippet:\n{response_snippet}\n")

                    except json.JSONDecodeError as json_err:
                        err_msg = f"Part {i} API success (200) but failed to decode JSON response."
                        response_snippet = response.text[:500]
                        print(f"ERROR: {err_msg} - {json_err}. Response Snippet: {response_snippet}")
                        errors_occurred.append(f"{err_msg} - {json_err} - Snippet: {response_snippet}")
                        all_results.append(f"## Part {i}/{total_parts} (ERROR: JSON Decode Failed)\n{err_msg}\nError: {json_err}\nResponse Snippet:\n{response_snippet}\n")
                    except Exception as e_parse:
                         err_msg = f"Part {i} Error parsing successful API response: {e_parse}"
                         response_snippet = response.text[:500]
                         print(f"ERROR: {err_msg} Response Snippet: {response_snippet}")
                         print(traceback.format_exc())
                         errors_occurred.append(f"{err_msg} - Snippet: {response_snippet}")
                         all_results.append(f"## Part {i}/{total_parts} (ERROR: Result Parsing Failed)\n{err_msg}\nResponse Snippet:\n{response_snippet}\n")

                # --- Handle API Errors ---
                else:
                    err_msg = f"Part {i} API Error {response.status_code} using {key_log_name}."
                    response_text_snippet = response.text[:500] if response.text else "(No response body)"
                    try:
                        error_details = response.json() # Try parsing as JSON first
                        err_msg += f"\nDetails: {error_details}"
                    except json.JSONDecodeError:
                        err_msg += f"\nResponse Text Snippet: {response_text_snippet}"
                    print(f"ERROR: {err_msg}")
                    errors_occurred.append(err_msg)
                    all_results.append(f"## Part {i}/{total_parts} (ERROR: API Call Failed {response.status_code})\n{err_msg}\n")
                    # No automatic retry or wait implemented here - could add if needed, but increases timeout risk

            # --- Handle Request Exceptions (Network, Timeout) ---
            except requests.exceptions.RequestException as req_err:
                 part_api_time = time.time() - part_start_time
                 err_msg = f"Part {i} Network/Request Error using {key_log_name} after {part_api_time:.2f}s: {req_err}"
                 print(f"ERROR: {err_msg}")
                 print(traceback.format_exc())
                 errors_occurred.append(err_msg)
                 all_results.append(f"## Part {i}/{total_parts} (ERROR: Request Failed)\n{err_msg}\n")
                 # Stop processing further parts if a network error occurs? Or continue? Currently continues.


            # Check for Netlify timeout approaching (heuristic) - This is unreliable!
            elapsed_total = time.time() - start_time_total
            # Netlify standard timeout is often 10s or 26s depending on plan/settings. Let's use 8s as a safety margin.
            if elapsed_total > 8.0 and i < total_parts:
                timeout_msg = f"Warning: Processing potentially timed out after {elapsed_total:.1f}s at part {i}/{total_parts}. Returning partial results."
                print(timeout_msg)
                errors_occurred.append(timeout_msg)
                all_results.append(f"## PROCESSING STOPPED (Timeout Risk)\n{timeout_msg}\n")
                break # Stop processing more parts

        # --- Combine Results ---
        final_result_text = "\n".join(all_results)
        total_duration = time.time() - start_time_total
        print(f"Finished processing all parts in {total_duration:.2f} seconds.")

        if errors_occurred:
            print(f"Completed with {len(errors_occurred)} error(s):")
            for err in errors_occurred: print(f"- {err}")
            # Maybe return a different status code or flag in response if errors occurred?
            # For simplicity, returning 200 but including errors in the result text.
            return jsonify({"result": final_result_text, "status": "Completed with errors."})
        else:
            return jsonify({"result": final_result_text, "status": "Completed successfully."})

    # --- Handle Outer Exceptions (Validation, Splitting) ---
    except ValueError as ve:
         print(f"Value Error: {ve}")
         print(traceback.format_exc())
         return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Unhandled exception in handle_process: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred."}), 500


# --- For Local Testing (optional) ---
# You can run this script directly using `python process.py`
# and test the endpoint using tools like curl or Postman
# Make sure to set dummy environment variables locally if you do this.
# Example: export KEY1='sk-...'
# if __name__ == '__main__':
#     print("Attempting to run Flask app locally on port 5000...")
#     if not VALID_API_KEYS:
#         print("\nWARNING: No API keys found in environment variables (KEY1-KEY4).")
#         print("Set them using 'export KEY1=...' before running locally for API calls to work.\n")
#     app.run(debug=True, port=5000) # debug=True provides auto-reload and more error details