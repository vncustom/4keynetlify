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
# Moved key loading inside the handler for per-request check & reporting
BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = 600 # 10 minutes timeout for API call (Netlify function timeout is shorter!)
EXPECTED_KEYS = ['KEY1', 'KEY2', 'KEY3', 'KEY4'] # Define expected keys

# --- Text Splitting Logic (Adapted from original - NO CHANGES HERE) ---
# ... (Keep the smart_split_by_words and split_text functions exactly as they were) ...

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

@app.route('/api/process', methods=['POST'])
def handle_process():
    """Handles the POST request to process text."""

    # --- Check Environment Variables ---
    loaded_keys = {}
    missing_keys = []
    key_status_messages = []

    for key_name in EXPECTED_KEYS:
        key_value = os.environ.get(key_name)
        if key_value and key_value.strip():
            loaded_keys[key_name] = key_value
            key_status_messages.append(f"{key_name}: Loaded OK")
            print(f"Found environment variable: {key_name}") # Log found keys
        else:
            missing_keys.append(key_name)
            key_status_messages.append(f"{key_name}: MISSING")
            print(f"WARNING: Missing environment variable: {key_name}") # Log missing keys

    VALID_API_KEYS = list(loaded_keys.values()) # Get list of actual key values
    key_load_summary = ", ".join(key_status_messages)

    # --- CRITICAL ERROR: No Keys Found ---
    if not VALID_API_KEYS:
        error_message = "Server configuration error: NO valid API keys (KEY1-KEY4) were found in environment variables. Cannot proceed."
        print(f"ERROR: {error_message}")
        # Return 500 Internal Server Error as it's a config issue
        return jsonify({"error": error_message, "key_status": key_load_summary}), 500

    num_available_keys = len(VALID_API_KEYS)
    print(f"Processing request with {num_available_keys} valid API key(s) found.")
    print(f"Key Load Summary: {key_load_summary}") # Log summary

    # --- Proceed with Processing ---
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
        if not model: return jsonify({"error": "Model not selected.", "key_status": key_load_summary}), 400
        if not split_method: return jsonify({"error": "Split method not selected.", "key_status": key_load_summary}), 400
        if not prompt: return jsonify({"error": "Prompt is required.", "key_status": key_load_summary}), 400
        if not text_to_process: return jsonify({"error": "Text to process is required.", "key_status": key_load_summary}), 400
        if split_method == "Theo số ký tự/từ" and (not isinstance(split_length, int) or split_length <= 0):
             return jsonify({"error": "Invalid split length for selected method.", "key_status": key_load_summary}), 400

        print(f"Received request: Model={model}, Lang={language}, Split={split_method}, Len={split_length if split_method == 'Theo số ký tự/từ' else 'N/A'}")

        # --- Split Text ---
        print("Splitting text...")
        parts = split_text(text_to_process, split_method, language, split_length)
        total_parts = len(parts)
        print(f"Text split into {total_parts} parts.")

        if total_parts == 0:
            return jsonify({"error": "Text could not be split into processable parts.", "key_status": key_load_summary}), 400

        # --- Process each part sequentially ---
        all_results = []
        errors_occurred = []
        start_time_total = time.time()

        for i, part_content in enumerate(parts, 1):
            part_start_time = time.time()
            print(f"--- Processing Part {i}/{total_parts} ---")

            # --- Select API Key (using the VALID_API_KEYS list loaded earlier) ---
            key_index = (i - 1) % num_available_keys
            current_api_key = VALID_API_KEYS[key_index]
            # Find original name for logging (e.g., KEY1, KEY3) - less efficient but better logging
            current_key_name = list(loaded_keys.keys())[list(loaded_keys.values()).index(current_api_key)]
            key_log_name = f"{current_key_name} (#{key_index + 1} in valid list)"
            print(f"Using {key_log_name}")

            # --- Prepare API Call ---
            full_prompt = f"{prompt}\n\n{part_content}"
            headers = {
                "Authorization": f"Bearer {current_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://YOUR_NETLIFY_SITE_NAME.netlify.app/", # Optional: Set your site URL
                "X-Title": "Netlify OpenRouter Client v1.1" # Version Bump
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}]
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
                # ... (Keep the response handling logic exactly as it was before) ...
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

            # --- Handle Request Exceptions (Network, Timeout) ---
            except requests.exceptions.RequestException as req_err:
                 part_api_time = time.time() - part_start_time
                 err_msg = f"Part {i} Network/Request Error using {key_log_name} after {part_api_time:.2f}s: {req_err}"
                 print(f"ERROR: {err_msg}")
                 print(traceback.format_exc())
                 errors_occurred.append(err_msg)
                 all_results.append(f"## Part {i}/{total_parts} (ERROR: Request Failed)\n{err_msg}\n")

            # Check for Netlify timeout approaching (heuristic)
            elapsed_total = time.time() - start_time_total
            if elapsed_total > 8.0 and i < total_parts: # Using 8s margin before typical 10s timeout
                timeout_msg = f"Warning: Processing potentially timed out after {elapsed_total:.1f}s at part {i}/{total_parts}. Returning partial results."
                print(timeout_msg)
                errors_occurred.append(timeout_msg)
                all_results.append(f"## PROCESSING STOPPED (Timeout Risk)\n{timeout_msg}\n")
                break # Stop processing more parts


        # --- Combine Results ---
        final_result_text = "\n".join(all_results)
        total_duration = time.time() - start_time_total
        print(f"Finished processing all parts in {total_duration:.2f} seconds.")

        response_payload = {
            "result": final_result_text,
            "key_status": key_load_summary # Include key status in response
        }

        if errors_occurred:
            print(f"Completed with {len(errors_occurred)} error(s).")
            response_payload["status"] = "Completed with errors."
            # Still return 200 OK, as processing attempted/partially completed
            return jsonify(response_payload), 200
        else:
            response_payload["status"] = "Completed successfully."
            return jsonify(response_payload), 200

    # --- Handle Outer Exceptions (Validation, Splitting) ---
    except ValueError as ve:
         print(f"Value Error: {ve}")
         print(traceback.format_exc())
         # Include key status even in validation errors if available
         return jsonify({"error": str(ve), "key_status": key_load_summary}), 400
    except Exception as e:
        print(f"Unhandled exception in handle_process: {e}")
        print(traceback.format_exc())
        # Include key status even in unexpected errors if available
        return jsonify({"error": "An unexpected server error occurred.", "key_status": key_load_summary}), 500

# --- Local Testing ---
# if __name__ == '__main__':
#     # ... (local testing code unchanged) ...