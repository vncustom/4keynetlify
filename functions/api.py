# -*- coding: utf-8 -*-
import json
import os
import re
import time
import traceback
from datetime import datetime
import threading
import requests
import uuid

# Global storage for ongoing processes
ACTIVE_PROCESSES = {}
RESULTS_DIR = "/tmp/openrouter_results"  # Netlify function temp directory for storing results

# Check if directory exists, if not create it
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper functions
def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def log_message(message, log_type="INFO"):
    print(f"[{get_timestamp()}][{log_type}] {message}")

# Load API keys from environment variables
def load_api_keys():
    keys = []
    key_status = []
    
    for i in range(1, 5):
        key_name = f"KEY{i}"
        key_value = os.getenv(key_name)
        
        if key_value and key_value.strip():
            keys.append(key_value.strip())
            key_status.append(f"✅ {key_name}: Đã tải thành công.")
        else:
            key_status.append(f"⚪ {key_name}: Trống hoặc không được đặt.")
    
    return keys, key_status

# Text splitting functions
def split_text_by_chapter(text):
    """Split text by chapter markers."""
    chapter_pattern = re.compile(
        r'(?:^|\n)\s*(第\s*\d+\s*章|Chương\s+\d+).*?(?=\n|\Z)',
        re.MULTILINE
    )
    
    # Find all matches
    matches = list(chapter_pattern.finditer(text))
    markers = [match.start(1) for match in matches]
    
    if not markers:
        # No chapter markers found, treat the whole text as one part
        return [text.strip()] if text.strip() else []
    
    parts = []
    num_markers = len(markers)
    
    for i in range(num_markers):
        # Start position is the beginning of the marker line found
        start_pos = markers[i]
        # End position is the start of the *next* marker line, or end of the entire text
        end_pos = markers[i+1] if i + 1 < num_markers else len(text)
        
        # Extract the content including the marker line itself
        part_content_raw = text[start_pos:end_pos]
        part_content_stripped = part_content_raw.strip()
        
        if part_content_stripped:
            parts.append(part_content_stripped)
    
    return parts

def split_text_by_characters(text, split_length):
    """Split text by character count."""
    parts = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + split_length, text_len)
        break_point = end
        
        if end < text_len:
            found_break = -1
            # Prioritize sentence ending or newline
            for bc in ".!?。\n":
                pos = text.rfind(bc, start, end)
                if pos > start and pos > found_break:
                    found_break = pos
            
            if found_break != -1:
                break_point = found_break + 1
            else:  # If no sentence end/newline, try space
                space_break = text.rfind(' ', start, end)
                if space_break > start:
                    break_point = space_break + 1
        
        chunk = text[start:break_point].strip()
        if chunk:
            parts.append(chunk)
        
        start = break_point
    
    return parts

def split_text_by_words(text, max_words):
    """Split English text by word count."""
    sentences = re.split(r'([.!?])\s*', text.strip())
    combined_sentences = []
    
    i = 0
    while i < len(sentences):
        sentence_text = sentences[i]
        i += 1
        delimiter = sentences[i] if i < len(sentences) and sentences[i] in ".!?" else ''
        if delimiter:
            i += 1
        if sentence_text and sentence_text.strip():
            combined_sentences.append(sentence_text.strip() + delimiter)
    
    chunks = []
    current_chunk_words = []
    current_word_count = 0
    
    for sentence in combined_sentences:
        if not sentence.strip():
            continue
        
        words_in_sentence = sentence.split()
        num_words = len(words_in_sentence)
        
        if num_words == 0:
            continue
        
        if num_words > max_words:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words, current_word_count = [], 0
            
            for j in range(0, num_words, max_words):
                chunk_part = " ".join(words_in_sentence[j:j+max_words])
                chunks.append(chunk_part)
        elif current_word_count + num_words <= max_words:
            current_chunk_words.extend(words_in_sentence)
            current_word_count += num_words
        else:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            current_chunk_words = words_in_sentence
            current_word_count = num_words
    
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    
    return [chunk for chunk in chunks if chunk and chunk.strip()]

# Main text processing function
def process_text(event_data, process_id):
    """Process text according to specified parameters."""
    # Get valid API keys
    api_keys, _ = load_api_keys()
    if not api_keys:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "No valid API keys found"})
        }
    
    # Extract parameters from the event
    language = event_data.get('language', 'Việt Nam')
    model = event_data.get('model', 'google/gemini-2.0-flash-lite-preview-02-05:free')
    split_method = event_data.get('splitMethod', 'chapter')
    split_length = event_data.get('splitLength', 15000)
    prompt_text = event_data.get('promptText', '')
    additional_text = event_data.get('additionalText', '')
    
    # Set up SSE connection
    process = ACTIVE_PROCESSES.get(process_id)
    if not process:
        return {
            "statusCode": 404,
            "body": json.dumps({"error": "Process not found"})
        }

    queue = process["queue"]
    
    # Update client with initial message
    queue.append({
        "type": "message",
        "outputType": "progress",
        "message": "Phase 1: Đang chia văn bản...",
        "clear": True
    })
    
    # Split text
    parts = []
    try:
        if split_method == "chapter":
            parts = split_text_by_chapter(additional_text)
        else:  # character/word count
            if language == "ENG":
                parts = split_text_by_words(additional_text, split_length)
            else:
                parts = split_text_by_characters(additional_text, split_length)
        
        if not parts:
            queue.append({
                "type": "error",
                "message": "Không có phần nào để xử lý sau khi chia văn bản (kết quả rỗng).",
                "fatal": True
            })
            return
        
        total_parts = len(parts)
        queue.append({
            "type": "message",
            "outputType": "progress",
            "message": f"✅ Chia thành công {total_parts} phần.",
            "clear": False
        })
    except Exception as e:
        queue.append({
            "type": "error",
            "message": f"Lỗi nghiêm trọng khi chia văn bản: {str(e)}",
            "fatal": True
        })
        queue.append({
            "type": "message",
            "outputType": "debug",
            "message": f"Exception during split_text: {traceback.format_exc()}",
            "clear": False
        })
        return
    
    # Prepare file for saving results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_model_name = model.replace('/', '_').replace(':', '-')
    result_file_path = os.path.join(RESULTS_DIR, f"result_{safe_model_name}_{timestamp}.txt")
    
    try:
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write("\n---\n\n")
        
        queue.append({
            "type": "message",
            "outputType": "completion",
            "message": f"Chuẩn bị lưu kết quả vào:\n{result_file_path}",
            "clear": True
        })
    except Exception as e:
        queue.append({
            "type": "error",
            "message": f"Không thể ghi thông tin ban đầu vào file kết quả: {str(e)}",
            "fatal": True
        })
        return
    
    # Process each part
    queue.append({
        "type": "message",
        "outputType": "progress",
        "message": f"Phase 2: Bắt đầu xử lý {len(parts)} phần...",
        "clear": False
    })
    
    error_occurred = False
    total_parts = len(parts)
    
    base_url = "https://openrouter.ai/api/v1"
    num_available_keys = len(api_keys)
    
    for i, part_content in enumerate(parts, 1):
        # Check if process should stop
        if process.get("should_stop", False):
            queue.append({
                "type": "message",
                "outputType": "completion",
                "message": f"⏩ Xử lý đã bị dừng theo yêu cầu ở phần {i}.",
                "clear": False
            })
            break
        
        # Update progress
        queue.append({
            "type": "progress",
            "current": i,
            "total": total_parts
        })
        
        current_part_success = False
        result_text_for_file = f"LỖI: Xử lý phần {i} không thành công hoặc không nhận được kết quả."
        
        try:
            # Select API Key
            key_index = (i - 1) % num_available_keys
            current_api_key = api_keys[key_index]
            key_log_name = f"Key #{key_index + 1}"
            
            queue.append({
                "type": "message",
                "outputType": "progress",
                "message": f"🔄 Đang gửi yêu cầu xử lý phần {i}/{total_parts} (sử dụng {key_log_name})...",
                "clear": True
            })
            
            # Combine prompt and current part content
            full_prompt = f"{prompt_text}\n\n{part_content}"
            
            headers = {
                "Authorization": f"Bearer {current_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://netlify.app/",
                "X-Title": "Netlify OpenRouter Client v1.0"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}]
            }
            
            # Make API Request
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=600
            )
            
            # Handle API Response
            if response.status_code == 200:
                try:
                    result_json = response.json()
                    
                    if not isinstance(result_json, dict):
                        raise ValueError(f"API response is not a valid JSON object. Type: {type(result_json)}")
                    
                    # Extract content
                    choices = result_json.get('choices')
                    if (isinstance(choices, list) and
                        len(choices) > 0 and
                        isinstance(choices[0], dict) and
                        'message' in choices[0] and
                        isinstance(choices[0]['message'], dict) and
                        'content' in choices[0]['message']):
                        
                        result_text = choices[0]['message']['content']
                        if result_text and isinstance(result_text, str) and result_text.strip():
                            result_text_for_file = result_text
                            
                            queue.append({
                                "type": "message",
                                "outputType": "result",
                                "message": f"Kết quả phần {i} (đầu):\n---\n{result_text[:1000]}...\n---",
                                "clear": True
                            })
                            
                            current_part_success = True
                        else:
                            content_type = type(result_text).__name__
                            is_empty_str = isinstance(result_text, str) and not result_text.strip()
                            result_text_for_file = f"Lỗi API: Phản hồi thành công (200) nhưng content trống hoặc không phải string (Type: {content_type}, EmptyStr: {is_empty_str}) cho phần {i}."
                            
                            queue.append({
                                "type": "error",
                                "message": result_text_for_file,
                                "fatal": False
                            })
                    else:
                        result_text_for_file = f"Lỗi API: Phản hồi thành công (200) nhưng cấu trúc JSON không hợp lệ hoặc thiếu trường 'content' cho phần {i}."
                        
                        queue.append({
                            "type": "error",
                            "message": result_text_for_file,
                            "fatal": False
                        })
                
                except Exception as e:
                    result_text_for_file = f"Lỗi không xác định khi xử lý response JSON thành công (200) cho phần {i}: {str(e)}"
                    
                    queue.append({
                        "type": "error",
                        "message": result_text_for_file,
                        "fatal": False
                    })
            
            else:
                error_message = f"Lỗi API {response.status_code} khi xử lý phần {i} (với {key_log_name})."
                
                try:
                    error_details = response.json()
                    error_message += f"\nDetails: {error_details}"
                except:
                    error_message += f"\nResponse Text Snippet: {response.text[:500]}..."
                
                result_text_for_file = error_message
                
                queue.append({
                    "type": "error",
                    "message": result_text_for_file,
                    "fatal": False
                })
                
                error_occurred = True
                
                # Specific handling for common errors
                if response.status_code == 429:
                    queue.append({
                        "type": "message",
                        "outputType": "progress",
                        "message": f"⏳ Bị giới hạn Rate Limit! Chờ 60 giây...",
                        "clear": False
                    })
                    
                    # Wait 60 seconds if rate limited
                    for _ in range(60):
                        if process.get("should_stop", False):
                            break
                        time.sleep(1)
            
            # Append Result/Error to File
            try:
                with open(result_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"## ")
                    f.write(f"{result_text_for_file}\n\n")
                
                if current_part_success:
                    queue.append({
                        "type": "message",
                        "outputType": "completion",
                        "message": f"✅ Đã hoàn thành và lưu phần {i}/{total_parts} vào file.",
                        "clear": True
                    })
                else:
                    queue.append({
                        "type": "message",
                        "outputType": "completion",
                        "message": f"⚠️ Đã ghi lỗi xử lý phần {i}/{total_parts} vào file.",
                        "clear": True
                    })
                    error_occurred = True
            
            except Exception as e:
                queue.append({
                    "type": "error",
                    "message": f"Lỗi nghiêm trọng khi ghi phần {i} vào file '{result_file_path}': {str(e)}",
                    "fatal": False
                })
                error_occurred = True
            
            # Wait before next request (if not rate limited)
            if i < total_parts and not process.get("should_stop", False) and response.status_code != 429:
                wait_time = 20
                
                queue.append({
                    "type": "message",
                    "outputType": "progress",
                    "message": f"⏳ Chờ {wait_time} giây trước khi gửi yêu cầu phần {i+1}/{total_parts}...",
                    "clear": False
                })
                
                for t in range(wait_time):
                    if process.get("should_stop", False):
                        break
                    
                    remaining_time = wait_time - t
                    queue.append({
                        "type": "message",
                        "outputType": "progress",
                        "message": f"⏳ Chờ {remaining_time} giây... (Phần {i+1}/{total_parts})",
                        "clear": True
                    })
                    time.sleep(1)
        
        except Exception as e:
            error_msg = f"Lỗi không xác định khi xử lý phần {i}: {str(e)}"
            
            queue.append({
                "type": "error",
                "message": error_msg,
                "fatal": False
            })
            
            error_occurred = True
            
            try:
                with open(result_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"## Phần {i}/{total_parts}\n### LỖI NGHIÊM TRỌNG KHI XỬ LÝ ###\n")
                    f.write(f"{error_msg}\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
                
                queue.append({
                    "type": "message",
                    "outputType": "completion",
                    "message": f"⚠️ Đã ghi lỗi nghiêm trọng phần {i}/{total_parts} vào file.",
                    "clear": True
                })
            except Exception as e_write:
                queue.append({
                    "type": "error",
                    "message": f"Lỗi khi ghi lỗi nghiêm trọng phần {i} vào file: {str(e_write)}",
                    "fatal": False
                })
    
    # Final status update
    final_message = ""
    
    if process.get("should_stop", False):
        final_message = f"⏩ Xử lý đã bị dừng. {i}/{total_parts} phần có thể đã được xử lý và lưu."
    elif error_occurred:
        final_message = f"❌ Hoàn thành với lỗi ở một số phần. Kết quả đã được ghi (bao gồm cả thông báo lỗi)."
    else:
        final_message = f"🎉 Hoàn tất xử lý và lưu thành công tất cả {total_parts} phần."
    
    queue.append({
        "type": "message",
        "outputType": "completion",
        "message": f"{final_message}\nFile kết quả:\n{result_file_path}",
        "clear": True
    })
    
    # Send completion signal
    queue.append({
        "type": "complete",
        "success": not error_occurred and not process.get("should_stop", False),
        "filePath": result_file_path
    })

# Netlify Function handlers
def check_keys_handler(event, context):
    """Handler for checking API keys."""
    keys, messages = load_api_keys()
    
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            "validKeys": keys,
            "messages": messages
        })
    }

def process_text_handler(event, context):
    """Handler for processing text."""
    try:
        body = json.loads(event["body"])
        
        # Generate a unique process ID
        process_id = str(uuid.uuid4())
        
        # Create a new process entry
        ACTIVE_PROCESSES[process_id] = {
            "queue": [],
            "should_stop": False,
            "start_time": datetime.now(),
            "data": body
        }
        
        # Start processing in a background thread
        thread = threading.Thread(
            target=process_text,
            args=(body, process_id)
        )
        thread.daemon = True
        thread.start()
        
        return {
            "statusCode": 200,
            "body": process_id
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        }

def process_updates_handler(event, context):
    """Handler for SSE updates from a process."""
    process_id = event.get("queryStringParameters", {}).get("id")
    
    if not process_id or process_id not in ACTIVE_PROCESSES:
        return {
            "statusCode": 404,
            "body": json.dumps({"error": "Process not found"})
        }
    
    process = ACTIVE_PROCESSES[process_id]
    queue = process["queue"]
    
    # Check if there are updates in the queue
    if queue:
        update = queue.pop(0)  # Get the oldest update
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            },
            "body": f"data: {json.dumps(update)}\n\n"
        }
    else:
        # No updates yet, return an empty response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            },
            "body": ": ping\n\n"  # Send a comment as heartbeat
        }

def stop_processing_handler(event, context):
    """Handler for stopping a process."""
    # Simply set the stop flag for all active processes
    for process_id in ACTIVE_PROCESSES:
        ACTIVE_PROCESSES[process_id]["should_stop"] = True
    
    return {
        "statusCode": 200,
        "body": json.dumps({"status": "stop_requested"})
    }

def load_results_handler(event, context):
    """Handler for loading the most recent results."""
    try:
        if not os.path.exists(RESULTS_DIR) or not os.path.isdir(RESULTS_DIR):
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "success": False,
                    "message": f"Results directory '{RESULTS_DIR}' does not exist."
                })
            }
        
        # Find files matching the pattern result_*_*.txt
        files_list = [f for f in os.listdir(RESULTS_DIR)
                     if f.startswith("result_") and f.endswith(".txt") and os.path.isfile(os.path.join(RESULTS_DIR, f))]
        
        if not files_list:
            return {
                "statusCode": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "success": False,
                    "message": f"No result files found in '{RESULTS_DIR}'."
                })
            }
        
        # Sort files by modification time (most recent first)
        files_list.sort(key=lambda f: os.path.getmtime(os.path.join(RESULTS_DIR, f)), reverse=True)
        latest_file = files_list[0]
        latest_file_path = os.path.join(RESULTS_DIR, latest_file)
        
        with open(latest_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "success": True,
                "fileName": latest_file,
                "filePath": latest_file_path,
                "content": content
            })
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        }

# Handler mapping for Netlify Functions
def handler(event, context):
    """Main handler that routes to specific function handlers."""
    path = event.get("path", "")
    
    if path.endswith("/check-keys"):
        return check_keys_handler(event, context)
    elif path.endswith("/process-text"):
        return process_text_handler(event, context)
    elif path.endswith("/process-updates"):
        return process_updates_handler(event, context)
    elif path.endswith("/stop-processing"):
        return stop_processing_handler(event, context)
    elif path.endswith("/load-results"):
        return load_results_handler(event, context)
    else:
        return {
            "statusCode": 404,
            "body": "Not found"
        }