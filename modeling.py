import os
import os
import json
from datetime import datetime
from typing import List, Dict, Union
import tiktoken
import time
# Updated imports for LangChain v0.2+
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import logging

# Add logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Move this function outside of the if __name__ block so it can be used by other functions
def initialize_llm() -> ChatOpenAI:
    """Initialize and return the Typhoon LLM client using LangChain's OpenAI integration."""
    try:
        # Load API key from config file
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        api_key = config.get('api_key')
        
        if not api_key:
            raise ValueError("API key not found in config.json")
            
    except FileNotFoundError:
        logger.error("config.json file not found. Please create it with your API key.")
        raise FileNotFoundError("config.json file not found")
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in config.json")
        raise ValueError("Invalid JSON format in config.json")
    
    return ChatOpenAI(
        model='typhoon-v2-70b-instruct',
        base_url='https://api.opentyphoon.ai/v1',
        api_key=api_key,
        temperature=0.7,  # Slightly lower temperature for more consistent style
        top_p=0.9,         # Moved from model_kwargs to explicit parameter
    )
def select_json_file(folder_path="data"):
    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return None

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in folder '{folder_path}'.")
        return None

    print("Available JSON files:")
    for index, file in enumerate(json_files, start=1):
        print(f"{index}: {file}")

    while True:
        choice = input("Enter the number of the JSON file you want to select: ").strip()
        if not choice.isdigit():
            print("Please enter a numeric value.")
            continue

        choice_int = int(choice)
        if 1 <= choice_int <= len(json_files):
            selected_file = os.path.join(folder_path, json_files[choice_int - 1])
            print(f"Selected file: {selected_file}")
            return selected_file
        else:
            print("Invalid choice. Please select a number from the list.")

def convert_unicode_to_thai(text: str) -> str:
    """Convert unicode-escaped Thai text to proper Thai characters."""
    # This handles common encoding issues with Thai text in exported Facebook data
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text




# --- RAG utilities import ---
from chat_rag import build_and_save_index, get_fewshot_qa_context

def load_and_process_messages(file_path: str, preprocess_for_eda: bool = False) -> Union[Dict, List]:
    """
    Load and process messages from JSON file with EDA preprocessing support
    
    Args:
        file_path: Path to the JSON file containing messages
        preprocess_for_eda: Whether to include EDA statistics preprocessing
        
    Returns:
        Dictionary with messages and statistics if preprocess_for_eda=True, 
        otherwise just the loaded data
    """
    try:
        logger.info(f"Loading messages from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract messages and participants
        if isinstance(data, dict):
            messages = data.get('messages', [])
            participants = data.get('participants', [])
        elif isinstance(data, list):
            messages = data
            participants = []
        else:
            logger.error("Invalid data format")
            return None

        # Extract participants (support both Messenger and Instagram format)
        if participants and isinstance(participants[0], dict):
            participants = [p.get('name', '') for p in participants]
        elif participants and isinstance(participants[0], str):
            participants = participants
        
        # Filter out empty messages and system messages
        filtered_messages = []
        empty_message_count = 0
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
                
            content = msg.get('content', '').strip()
            
            # Count empty messages
            if not content:
                empty_message_count += 1
                continue
            
            # Enhanced filtering for system messages and attachments
            if (content.startswith(('คุณเริ่มการโทร', 'การโทรด้วยเสียงสิ้นสุด', 'You sent an attachment', 
                                   'Thanayot sent an attachment', 'แสดงความรู้สึก', 'ถูกใจข้อความ',
                                   'wasn\'t notified about this message', 'เปลี่ยนธีม')) or
                'แสดงความรู้สึก' in content or 'ต่อข้อความของคุณ' in content or
                'sent an attachment' in content or 'การโทรด้วยเสียงสิ้นสุดลงแล้ว' in content or
                'เริ่มการโทรด้วยเสียง' in content or 'คุณไม่ได้รับสายการโทรด้วยเสียง' in content):
                continue
            
            # Skip URLs and link messages
            if content.startswith(('http', 'https', 'www.')):
                continue

            # Try both Messenger and Instagram fields for timestamp
            timestamp = msg.get('timestamp_ms', 0) or msg.get('timestamp', 0)
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except Exception:
                        timestamp = timestamp
                else:
                    timestamp = datetime.fromtimestamp(int(timestamp) / 1000) if int(timestamp) > 1000000000000 else datetime.fromtimestamp(int(timestamp))
            
            processed_msg = {
                'sender': msg.get('sender_name', msg.get('sender', '')),
                'content': content,
                'timestamp': timestamp,
                'type': msg.get('type', ''),
                'photos': msg.get('photos', []),
                'videos': msg.get('videos', []),
                'reactions': msg.get('reactions', [])
            }
            
            # Convert unicode to Thai if needed
            processed_msg['sender'] = convert_unicode_to_thai(processed_msg['sender'])
            processed_msg['content'] = convert_unicode_to_thai(processed_msg['content'])
            
            filtered_messages.append(processed_msg)
        
        logger.info(f"Processed {len(filtered_messages)} valid messages from {len(messages)} total messages")
        logger.info(f"Skipped {empty_message_count} empty messages")
        
        # If no valid messages found, create sample data
        if not filtered_messages:
            logger.warning("No valid messages found. Creating sample data for demonstration.")
            sample_messages = create_sample_thai_messages(participants)
            filtered_messages = sample_messages

        # Sort messages by timestamp (oldest to newest)
        filtered_messages.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)

        # Save to memory.json in the expected format for RAG
        final_data = {
            "participants": participants,
            "messages": filtered_messages
        }
        
        # Always return consistent format
        result = {
            'messages': filtered_messages,
            'participants': participants
        }
        
        if preprocess_for_eda:
            # Generate EDA statistics - this would be implemented in thai_chat_analysis.py
            try:
                from eda_data import ThaiChatAnalyzer
                analyzer = ThaiChatAnalyzer()
                df = analyzer.preprocess_messages(result)
                stats = analyzer.create_basic_statistics(df)
                word_analysis = analyzer.analyze_word_patterns(df)
                
                result['eda_data'] = {
                    'dataframe': df,
                    'statistics': stats,
                    'word_analysis': word_analysis
                }
            except Exception as e:
                logger.warning(f"Could not generate EDA statistics: {e}")

        # For backward compatibility, also save to memory.json
        with open('memory.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, default=str, ensure_ascii=False, indent=2)

        # --- RAG: Build embeddings and FAISS index for retrieval ---
        try:
            build_and_save_index()
        except Exception as e:
            logger.warning(f"Could not build RAG index: {e}")
        
        return result if preprocess_for_eda else filtered_messages
            
    except Exception as e:
        logger.error(f"Error loading messages: {e}")
        return None

def create_sample_thai_messages(participants):
    """Create sample Thai messages for demonstration"""
    if not participants:
        participants = ["User1", "User2"]
    
    sample_messages = [
        {
            "sender": participants[0] if len(participants) > 0 else "User1",
            "content": "สวัสดีครับ เป็นยังไงบ้าง",
            "timestamp": "2024-01-01T10:00:00",
            "type": "Generic"
        },
        {
            "sender": participants[1] if len(participants) > 1 else "User2", 
            "content": "สบายดีค่ะ วันนี้ทำอะไรหรอคะ",
            "timestamp": "2024-01-01T10:01:00",
            "type": "Generic"
        },
        {
            "sender": participants[0] if len(participants) > 0 else "User1",
            "content": "ไปทำงานมาครับ เหนื่อยมากเลย",
            "timestamp": "2024-01-01T10:02:00",
            "type": "Generic"
        },
        {
            "sender": participants[1] if len(participants) > 1 else "User2",
            "content": "หิวข้าวแล้วนะ อยากกินอะไรดี",
            "timestamp": "2024-01-01T10:03:00",
            "type": "Generic"
        },
        {
            "sender": participants[0] if len(participants) > 0 else "User1",
            "content": "ไปกินข้าวกันมั้ย ผมรู้จักร้านอร่อยๆ",
            "timestamp": "2024-01-01T10:04:00",
            "type": "Generic"
        }
    ]
    
    return sample_messages

# --- RAG utilities import ---
from chat_rag import build_and_save_index, get_fewshot_qa_context

def load_and_process_messages(file_path: str) -> List[Dict]:
    """
    Load Instagram messages from JSON file and process them into a structured format.
    Returns list of processed messages.
    Also triggers RAG index build for downstream retrieval.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract participants (support both Messenger and Instagram format)
        participants = []
        if data.get('participants') and isinstance(data['participants'][0], dict):
            participants = [p.get('name', '') for p in data.get('participants', [])]
        elif data.get('participants') and isinstance(data['participants'][0], str):
            participants = data['participants']

        processed_messages = []
        for msg in data.get('messages', []):
            if not isinstance(msg, dict):
                continue
                
            # Skip messages with specific content patterns
            content = msg.get('content', '')
            if any(pattern in content for pattern in [
                'แสดงความรู้สึก',
                'ต่อข้อความของคุณ',
                'การโทรด้วยเสียงสิ้นสุดลงแล้ว',
                'เริ่มการโทรด้วยเสียง',
                'คุณไม่ได้รับสายการโทรด้วยเสียง'
                'sent an attachment',
                "wasn't notified about this message because they're in quiet mode."
            ]):
                continue
                
            # Try both Messenger and Instagram fields
            timestamp = msg.get('timestamp_ms', 0) or msg.get('timestamp', 0)
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except Exception:
                        timestamp = timestamp
                else:
                    timestamp = datetime.fromtimestamp(int(timestamp) / 1000) if int(timestamp) > 1000000000000 else datetime.fromtimestamp(int(timestamp))
            processed_msg = {
                'sender': msg.get('sender_name', msg.get('sender', '')),
                'content': msg.get('content', ''),
                'timestamp': timestamp,
                'type': msg.get('type', ''),
                'photos': msg.get('photos', []),
                'videos': msg.get('videos', []),
                'reactions': msg.get('reactions', [])
            }
            processed_msg['sender'] = convert_unicode_to_thai(processed_msg['sender'])
            if processed_msg['content']:
                processed_msg['content'] = convert_unicode_to_thai(processed_msg['content'])
            processed_messages.append(processed_msg)

        # Sort messages by timestamp (oldest to newest)
        processed_messages.sort(key=lambda x: x['timestamp'])

        # Save to memory.json in the expected format for RAG
        final_data = {
            "participants": participants,
            "messages": processed_messages
        }
        with open('memory.json', 'w', encoding='utf-8') as f:
            json.dump(final_data, f, default=str, ensure_ascii=False, indent=2)

        # --- RAG: Build embeddings and FAISS index for retrieval ---
        build_and_save_index()

        return processed_messages
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def chunk_messages_into_qa(model_name: str = 'default', chunk_size: int = 1000, overlap: int = 100, limit: int = None) -> List[Dict]:
    """
    Convert memory.json content into Q/A format chunks for LLM processing.
    Each chunk will be approximately chunk_size tokens with overlap between chunks.
    
    Args:
        model_name: Name of the model directory
        chunk_size: Size of each chunk in tokens
        overlap: Number of tokens to overlap between chunks
        limit: Optional limit to the number of chunks processed (None = process all)
    
    Returns:
        List of processed chunks
    """
    json_path = f'model/{model_name}/memory.json'
    try:
        # Load the memory file
        with open(json_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        
        messages = memory_data.get('messages', [])
        participants = memory_data.get('participants', [])
        
        if not messages:
            print("No messages found in memory file.")
            return []
            
        # Initialize tokenizer once
        enc = tiktoken.get_encoding("cl100k_base")
        
        # Pre-process messages to reduce repeated operations and filter out more unwanted content
        print("Formatting conversation...")
        conversation = []
        for msg in messages:
            if msg.get('content'):
                content = msg.get('content')
                # Enhanced filtering during chunking
                if ('แสดงความรู้สึก' in content or 
                    'ต่อข้อความของคุณ' in content or 
                    'sent an attachment' in content or
                    'การโทรด้วยเสียงสิ้นสุดลงแล้ว' in content or
                    'เริ่มการโทรด้วยเสียง' in content or
                    content.startswith('https://') or 
                    'http://' in content or
                    content.isdigit()):  # Filter out pure number messages
                    continue
                    
                formatted_msg = f"{msg.get('sender', 'Unknown')}: {msg.get('content', '')}"
                conversation.append(formatted_msg)
        
        # Join all messages into a single text
        full_text = "\n".join(conversation)
        
        # Tokenize the full text (this is a bottleneck)
        print("Tokenizing text...")
        tokens = enc.encode(full_text)
        total_tokens = len(tokens)
        print(f"Total tokens: {total_tokens}")
        
        # Prepare chunks
        chunks = []
        start_idx = 0
        chunk_count = 0
        
        # Fix infinite loop by ensuring start_idx always advances
        # and adding a safety check
        max_iterations = (total_tokens // (chunk_size - overlap)) + 2  # Add some buffer
        iteration = 0
        
        print(f"Creating chunks with size={chunk_size}, overlap={overlap}...")
        while start_idx < total_tokens and iteration < max_iterations:
            iteration += 1
            
            # Progress indicator
            if chunk_count % 10 == 0:
                print(f"Processing chunk {chunk_count}, position {start_idx}/{total_tokens}")
                
            # Determine end index for current chunk
            end_idx = min(start_idx + chunk_size, total_tokens)
            
            # Safety check - ensure we're making progress
            if end_idx <= start_idx:
                print(f"Warning: No progress made in chunking at position {start_idx}")
                break
                
            # Extract token subset and decode back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = enc.decode(chunk_tokens)
            
            # Format as Q/A with simpler join operation
            participants_str = ", ".join(participants)
            chunk = {
                "question": f"Summarize this conversation segment between {participants_str}:",
                "answer": chunk_text,
                "token_count": len(chunk_tokens),
                "start_position": start_idx,
                "end_position": end_idx
            }
            
            chunks.append(chunk)
            chunk_count += 1
            
            # Check if we've reached the limit
            if limit is not None and chunk_count >= limit:
                print(f"Reached chunk limit of {limit}, stopping")
                break
            
            # Move to next chunk position with overlap
            # Ensure we always make forward progress to avoid infinite loop
            prev_start = start_idx
            start_idx = end_idx - overlap
            
            # If we didn't advance (can happen with very small chunk_size and large overlap)
            if start_idx <= prev_start:
                start_idx = prev_start + 1  # Force at least one token of progress
        
        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached, chunking may be incomplete")
        
        # Save chunks to file with progress indicator
        print("Saving chunks to file...")
        with open(f'model/{model_name}/chunks/memory_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        return chunks
        
    except Exception as e:
        print(f"Error chunking messages: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def summarize_chunks_with_typhoon(chunks, llm=None, model_name='default'):
    """
    Summarize chunks using Typhoon via LangChain and create a memory vectorstore.
    Ensures summaries are always generated and saved, with robust error handling and logging.
    Args:
        chunks: List of chunks containing conversation segments
        llm: Optional LangChain LLM instance
        model_name: Name of the model directory for saving summaries
    Returns:
        Tuple of (summaries, vectorstore)
    """
    print("Summarizing chunks with Typhoon...")
    if not chunks or len(chunks) == 0:
        print("No chunks provided for summarization.")
        # Always write an empty list to summaries file
        model_dir = f'model/{model_name}'
        os.makedirs(model_dir, exist_ok=True)
        with open(f'{model_dir}/memory_summaries.json', 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return [], None

    if llm is None:
        llm = initialize_llm()

    # Enhanced map prompt for summarization (Thai, relationship-agnostic, memory-focused)
    map_prompt_template = """
วิเคราะห์ส่วนของบทสนทนาต่อไปนี้และสกัดข้อมูลสำคัญ:

1. ชื่อและคำเรียกขาน:
- ชื่อจริง และชื่อเล่นที่สำคัญที่สุด 2 ชื่อที่ใช้เรียกกันบ่อยที่สุด
- รูปแบบการเรียกขานที่ใช้บ่อยที่สุด
- คำลงท้ายที่ใช้บ่อย

2. ความสัมพันธ์และบริบท:
- ประเภทความสัมพันธ์
- ระดับความสนิทและความเป็นทางการ
- บทบาทในความสัมพันธ์

3. รูปแบบการสื่อสาร:
- สไตล์การพูดคุย (เป็นทางการ/กันเอง)
- อารมณ์และน้ำเสียงที่ใช้
- มุกหรือคำพูดที่เป็นเอกลักษณ์

4. ประสบการณ์และความทรงจำร่วม:
- เหตุการณ์หรือประสบการณ์ที่ถูกอ้างถึง
- ความทรงจำร่วมที่สำคัญ
- เรื่องราวหรือประวัติที่ถูกพูดถึงบ่อย

บทสนทนา:
{text}

สรุปข้อมูลที่สกัดได้:
"""
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # Enhanced combine prompt for overall memory profile (Thai)
    combine_prompt_template = """
จากความทรงจำและบริบทที่สกัดจากแต่ละส่วนของบทสนทนา สร้างโปรไฟล์ความทรงจำที่ครอบคลุมซึ่งสะท้อนถึงรูปแบบการสื่อสาร ความสัมพันธ์ และรายละเอียดสำคัญที่ควรจดจำเพื่อให้ AI สามารถสนทนาได้อย่างต่อเนื่องและเป็นธรรมชาติในอนาคต

ไม่ต้องกล่าวถึงว่านี่คือการวิเคราะห์ แต่ให้สรุปเหมือนกำลังสร้างความทรงจำระยะยาว โดยเน้นที่:
- วิธีที่ผู้เข้าร่วมแต่ละคนปฏิสัมพันธ์กันในสถานการณ์ต่าง ๆ
- รูปแบบการสื่อสาร อารมณ์ และประวัติร่วมที่เกิดขึ้นซ้ำ
- ข้อเท็จจริง ความชอบ หรือรายละเอียดส่วนตัวที่ควรจดจำ
- สัญญาณหรือความเข้าใจโดยนัยที่มีผลต่อการสนทนา

ความทรงจำที่สกัดจากแต่ละส่วน:
{text}

โปรไฟล์ความทรงจำโดยรวม:
"""
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    # Initialize the summarization chain
    try:
        summary_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
    except Exception as e:
        print(f"Error initializing summarization chain: {str(e)}")
        summary_chain = None

    # Prepare documents for summarization
    documents = []
    for idx, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk["answer"],
            metadata={
                "index": idx,
                "question": chunk["question"],
                "token_count": chunk["token_count"],
                "positions": f"{chunk['start_position']}-{chunk['end_position']}"
            }
        )
        documents.append(doc)

    batch_size = 5
    all_summaries = []
    failed_batches = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
        try:
            if summary_chain is not None:
                batch_result = summary_chain.run(batch)
            else:
                batch_result = "[ERROR: Summarization chain not initialized]"
            all_summaries.append({
                "batch": i//batch_size + 1,
                "summary": batch_result,
                "chunk_indexes": list(range(i, min(i+batch_size, len(documents))))
            })
            print(f"✓ Completed batch {i//batch_size + 1}")
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            all_summaries.append({
                "batch": i//batch_size + 1,
                "summary": f"Error: {str(e)}",
                "chunk_indexes": list(range(i, min(i+batch_size, len(documents))))
            })
            failed_batches += 1

    # Save summaries to file (always write, even if empty or error)
    model_dir = f'model/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    try:
        with open(f'{model_dir}/memory_summaries.json', 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_summaries)} summaries to {model_dir}/memory_summaries.json")
    except Exception as e:
        print(f"Error saving memory summaries: {str(e)}")

    # Create memory vectorstore with enhanced documents
    print("Creating memory vectorstore...")
    enhanced_docs = []
    for summary in all_summaries:
        batch_summary = summary["summary"]
        for idx in summary["chunk_indexes"]:
            if idx < len(documents):
                doc = Document(
                    page_content=f"CONVERSATION:\n{documents[idx].page_content}\n\nSUMMARY:\n{batch_summary}",
                    metadata=documents[idx].metadata
                )
                enhanced_docs.append(doc)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(enhanced_docs, embeddings)
        vectorstore.save_local(f"{model_dir}/memory_vectorstore")
        print("Memory vectorstore created and saved")
        return all_summaries, vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        import traceback
        traceback.print_exc()
        return all_summaries, None

def create_conversational_memory(chunks, llm=None, model_name='default'):
    """
    Create a conversational memory system that includes both summaries
    and retrieval capabilities for natural conversation.
    
    Returns:
        A configured ConversationalRetrievalChain
    """
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    
    # First summarize and create the vectorstore
    summaries, vectorstore = summarize_chunks_with_typhoon(chunks, llm, model_name=model_name)
    
    if not vectorstore:
        print("Unable to create vectorstore. Using text-only memory.")
        return None
    
    # Initialize LLM if not provided
    if llm is None:
        llm = initialize_llm()
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Enhanced QA prompt for conversation (Thai, relationship-agnostic, memory-focused)
    qa_template = """
คุณคือ AI ที่เป็นคู่สนทนาตัวจริงในบทสนทนานี้ สามารถดึงและใช้ความทรงจำจากการสนทนาในอดีตทั้งหมดได้ ไม่ว่าความสัมพันธ์จะเป็นแบบใด (เพื่อน ครอบครัว คนรัก เพื่อนร่วมงาน ฯลฯ)

- ตอบกลับในลักษณะที่เหมาะสมกับความสัมพันธ์และสไตล์การสื่อสารที่ปรากฏในความทรงจำ
- ใช้ประสบการณ์ร่วม มุกส่วนตัว รายละเอียดส่วนตัว และอารมณ์ที่เกี่ยวข้องอย่างเหมาะสม
- ให้คำตอบที่สะท้อนรูปแบบการปฏิสัมพันธ์และบรรยากาศของบทสนทนา
- อย่ากล่าวถึงว่าคุณเป็น AI หรือกำลังใช้ความทรงจำ ให้ตอบเหมือนมนุษย์ที่มีความต่อเนื่องของประสบการณ์

ความทรงจำและบริบทที่เกี่ยวข้อง:
{context}

ประวัติการแชทล่าสุด:
{chat_history}

คู่สนทนา: {question}
คุณ:
"""
    QA_PROMPT = PromptTemplate(
        template=qa_template, 
        input_variables=["chat_history", "context", "question"]
    )
    
    # Create retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def save_chat_history_to_jsonl(conversation_memory, output_path="model/chat_history.jsonl"):
    """
    Extract and save chat history from a conversation memory chain to JSONL format
    that can be easily loaded back into LangChain.
    
    Args:
        conversation_memory: The ConversationalRetrievalChain containing memory
        output_path: Path to save the JSONL file
    """
    try:
        # Extract messages from conversation memory
        messages = conversation_memory.memory.chat_memory.messages
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write messages to JSONL file
        with open(output_path, 'w', encoding='utf-8') as f:
            for msg in messages:
                # Format message for LangChain compatibility
                message_entry = {
                    "type": msg.type,  # human or ai
                    "data": {
                        "content": msg.content,
                        "additional_kwargs": msg.additional_kwargs,
                        "example": False
                    }
                }
                f.write(json.dumps(message_entry, ensure_ascii=False) + '\n')
        
        print(f"Chat history saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")
        return False

def load_chat_history_from_jsonl(file_path="model/chat_history.jsonl"):
    """
    Load chat history from a JSONL file into a format that can be used by LangChain.
    
    Args:
        file_path: Path to the JSONL file containing chat history
        
    Returns:
        A list of message objects that can be added to a ConversationBufferMemory
    """
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    
    try:
        messages = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                
                # Convert to appropriate message type
                if entry["type"] == "human":
                    msg = HumanMessage(content=entry["data"]["content"])
                elif entry["type"] == "ai":
                    msg = AIMessage(content=entry["data"]["content"])
                elif entry["type"] == "system":
                    msg = SystemMessage(content=entry["data"]["content"])
                else:
                    continue
                    
                messages.append(msg)
                
        print(f"Loaded {len(messages)} messages from chat history")
        return messages
    except Exception as e:
        print(f"Error loading chat history: {str(e)}")
        return []

def create_conversation_with_history(history_path="model/chat_history.jsonl"):
    """
    Create a conversation chain with pre-loaded chat history.
    
    Args:
        history_path: Path to JSONL file with chat history
        
    Returns:
        A ConversationalRetrievalChain with loaded history
    """
    # Load the vectorstore
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("model/memory_vectorstore", embeddings, allow_dangerous_deserialization=True)
        
        # Initialize LLM
        llm = initialize_llm()
        
        # Create memory with loaded history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Load existing chat history
        chat_history = load_chat_history_from_jsonl(history_path)
        for message in chat_history:
            memory.chat_memory.add_message(message)
        
        # Create custom prompt for natural, friendly responses
        qa_template = """
        You are simulating a conversation between close friends. 
        You have access to memories of your past conversations with this friend.
        Use these memories to respond naturally, maintaining the tone and style 
        of your previous interactions.
        
        Remember personal details, inside jokes, and the way you typically communicate.
        Don't explicitly reference that you're using "memories" - just incorporate
        the information naturally as a friend would.
        
        Chat History:
        {chat_history}
        
        Context from past conversations:
        {context}
        
        Friend: {question}
        You:
        """
        
        QA_PROMPT = PromptTemplate(
            template=qa_template, 
            input_variables=["chat_history", "context", "question"]
        )
        
        # Create retrieval chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=True
        )
        
        print(f"Conversation created with {len(chat_history)} loaded messages")
        return conversation_chain
        
    except Exception as e:
        print(f"Error creating conversation with history: {str(e)}")
        return None
    
def process_participant_model(model_name, participant=None, chunk_limit=None, chunk_size=5):
    """
    Process chunks and create memory summaries for a specific participant and model.
    This is used during setup to create a cloned model for a specific participant.
    
    Args:
        model_name: Name of the model directory
        participant: Name of the participant to clone (optional)
        chunk_limit: Optional limit to the number of chunks processed
        chunk_size: Size of chunks (1-10 scale, where 1=small, 10=large). Affects chunk sizes and processing.
        
    Returns:
        Dict with process status and results
    """
    try:
        print(f"Processing model {model_name} for participant {participant}")
        
        # Get reference to global processing_status from app.py if available
        import sys
        if 'app' in sys.modules:
            from app import processing_status
            has_global_status = True
        else:
            has_global_status = False
        
        # Track progress for real-time updates
        progress = {
            "status": "processing",
            "current_step": "validation",
            "total_steps": 5,
            "step": 1,
            "message": "Validating memory data...",
            "chunks_processed": 0,
            "total_chunks": 0,
            "batches_processed": 0,
            "total_batches": 0
        }
        
        # Update global status if available
        def update_global_status():
            if has_global_status and model_name in processing_status:
                processing_status[model_name]['progress'] = progress
                processing_status[model_name]['status'] = progress['status']
        
        update_global_status()
        
        # Check if memory.json exists and validate its structure
        memory_path = f'model/{model_name}/memory.json'
        if not os.path.exists(memory_path):
            return {"status": "error", "message": f"Memory file not found at {memory_path}. Please upload and process a chat file first."}
        
        # Validate memory.json structure
        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
                
            if not isinstance(memory_data, dict) or 'messages' not in memory_data:
                return {"status": "error", "message": "Invalid memory.json format. Missing 'messages' key."}
                
            messages = memory_data.get('messages', [])
            if not messages:
                return {"status": "error", "message": "No messages found in memory.json"}
                
            progress["message"] = f"Found {len(messages)} messages in memory file"
            update_global_status()
            
        except Exception as e:
            return {"status": "error", "message": f"Error reading memory file: {str(e)}"}
        
        progress["step"] = 2
        progress["current_step"] = "chunking"
        progress["message"] = "Analyzing message data and determining optimal chunk size..."
        update_global_status()

        # Calculate token count of the entire message file to determine appropriate chunking
        with open(memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
            all_messages = memory_data.get('messages', [])
        
        # Get total token count using tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        total_text = ""
        for msg in all_messages:
            content = msg.get('content', '')
            if content:
                total_text += content + "\n"
        
        total_tokens = len(encoding.encode(total_text))
        
        # Map the chunk_size parameter (1-10) to actual token counts
        # Calculate a sensible chunk size based on total tokens and user preference
        # Small values (1-3) create many small chunks, large values (8-10) create fewer large chunks
        min_chunk_size = 800   # Minimum chunk size (for chunk_size=1)
        max_chunk_size = 3500  # Maximum chunk size (for chunk_size=10)
        
        # Scale chunk size based on user preference (chunk_size parameter)
        actual_chunk_size = min_chunk_size + ((max_chunk_size - min_chunk_size) * (chunk_size - 1) / 9)
        actual_chunk_size = int(actual_chunk_size)
        
        # Update progress with information about token count and chunking
        progress["message"] = f"Chunking messages into Q/A format... (Total tokens: {total_tokens}, Chunk size: {actual_chunk_size})"
        update_global_status()  # Update global status
        
        chunks = chunk_messages_into_qa(model_name=model_name, chunk_size=actual_chunk_size, limit=chunk_limit)
        progress["chunks_processed"] = len(chunks)
        progress["total_chunks"] = len(chunks)
        progress["step"] = 2
        progress["current_step"] = "summarizing"
        progress["message"] = f"Generating chunk summaries for {len(chunks)} chunks..."
        update_global_status()  # Update global status
        
        if not chunks:
            return {"status": "error", "message": "No chunks were created"}
        
        # Initialize LLM
        llm = initialize_llm()
        
        # Create paths for model-specific outputs
        model_dir = f'model/{model_name}'
        chunks_dir = f'{model_dir}/chunks'
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Save chunks to model-specific directory
        with open(f'{chunks_dir}/memory_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        # First step: Generate summaries    
        # Prepare documents for summarization
        documents = []
        for idx, chunk in enumerate(chunks):
            # Create a Document object for LangChain
            doc = Document(
                page_content=chunk["answer"],
                metadata={
                    "index": idx,
                    "question": chunk["question"],
                    "token_count": chunk["token_count"],
                    "positions": f"{chunk['start_position']}-{chunk['end_position']}"
                }
            )
            documents.append(doc)
        
        # Create map and combine prompts
        map_prompt_template = """
วิเคราะห์ส่วนของบทสนทนาต่อไปนี้และสกัดข้อมูลสำคัญ:

1. ชื่อและคำเรียกขาน:
- ชื่อจริง และชื่อเล่นที่สำคัญที่สุด 2 ชื่อที่ใช้เรียกกันบ่อยที่สุด
- รูปแบบการเรียกขานที่ใช้บ่อยที่สุด
- คำลงท้ายที่ใช้บ่อย

2. ความสัมพันธ์และบริบท:
- ประเภทความสัมพันธ์
- ระดับความสนิทและความเป็นทางการ
- บทบาทในความสัมพันธ์

3. รูปแบบการสื่อสาร:
- สไตล์การพูดคุย (เป็นทางการ/กันเอง)
- อารมณ์และน้ำเสียงที่ใช้
- มุกหรือคำพูดที่เป็นเอกลักษณ์

4. ประสบการณ์และความทรงจำร่วม:
- เหตุการณ์หรือประสบการณ์ที่ถูกอ้างถึง
- ความทรงจำร่วมที่สำคัญ
- เรื่องราวหรือประวัติที่ถูกพูดถึงบ่อย

บทสนทนา:
{text}

สรุปข้อมูลที่สกัดได้:
"""
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        
        combine_prompt_template = """
จากการสรุปส่วนต่างๆ ของบทสนทนา สร้างภาพรวมของความสัมพันธ์และการสื่อสารระหว่างคู่สนทนา
โดยไม่ต้องกล่าวถึงว่านี่คือการวิเคราะห์ แต่ให้สรุปว่าบุคคลเหล่านี้สื่อสารกันอย่างไร

บทสรุปส่วนต่างๆ:
{text}

ภาพรวมของความสัมพันธ์:
"""
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        # Initialize the summarization chain
        summary_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
          # Process in smaller batches to avoid context limits
        batch_size = 5
        all_summaries = []
        total_batches = (len(documents)-1)//batch_size + 1
        progress["total_batches"] = total_batches
        update_global_status()  # Update global status
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = i//batch_size + 1
            progress["message"] = f"Processing batch {batch_num}/{total_batches}..."
            print(f"Processing batch {batch_num}/{total_batches}...")
            update_global_status()  # Update global status before batch processing
            
            try:
                # Run the summary chain on this batch
                batch_result = summary_chain.run(batch)
                all_summaries.append({
                    "batch": batch_num,
                    "summary": batch_result,
                    "chunk_indexes": list(range(i, min(i+batch_size, len(documents))))
                })
                progress["batches_processed"] = batch_num
                update_global_status()  # Update after each batch
                print(f"✓ Completed batch {batch_num}")
            except Exception as e:
                print(f"Error processing batch {batch_num}: {str(e)}")
                all_summaries.append({
                    "batch": batch_num,
                    "summary": f"Error: {str(e)}",
                    "chunk_indexes": list(range(i, min(i+batch_size, len(documents))))
                })
                progress["batches_processed"] = batch_num
                update_global_status()  # Update after error
          # Prepare to save summaries
        progress["step"] = 3
        progress["current_step"] = "saving_summaries"
        progress["message"] = "Saving memory summaries..."
        update_global_status()  # Update status
        
        # Create memory vectorstore with enhanced documents
        progress["step"] = 4
        progress["current_step"] = "creating_vectorstore"
        progress["message"] = "Creating memory vectorstore..."
        print("Creating memory vectorstore...")
        enhanced_docs = []
        
        # Add summaries to documents
        for summary in all_summaries:
            batch_summary = summary["summary"]
            for idx in summary["chunk_indexes"]:
                if idx < len(documents):
                    # Create a new document with enhanced content
                    doc = Document(
                        page_content=f"CONVERSATION:\n{documents[idx].page_content}\n\nSUMMARY:\n{batch_summary}",
                        metadata=documents[idx].metadata
                    )
                    enhanced_docs.append(doc)
        
        try:
            # Create vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(enhanced_docs, embeddings)
            
            # Save vectorstore
            vectorstore.save_local(f"{model_dir}/memory_vectorstore")
            print("Memory vectorstore created and saved")
            
            progress["status"] = "complete"
            progress["message"] = "Setup complete! Your AI clone is ready."
            return {
                "status": "success", 
                "message": "Processing complete",
                "summaries": len(all_summaries),
                "chunks": len(chunks),
                "documents": len(enhanced_docs)
            }
        except Exception as e:
            print(f"Error creating vectorstore: {str(e)}")
            import traceback
            traceback.print_exc()
            progress["status"] = "error"
            progress["message"] = f"Error creating vectorstore: {str(e)}"
            return {"status": "error", "message": f"Error creating vectorstore: {str(e)}"}
    except Exception as e:
        print(f"Error in process_participant_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    
if __name__ == "__main__":
    selected_file = select_json_file("data")
    if selected_file:
        print(f"Processing file: {selected_file}")
        messages = load_and_process_messages(selected_file)
        
        # Create directories only once
        os.makedirs('model/chunks', exist_ok=True)
        
        # Move memory.json directly to target path
        os.replace('memory.json', 'model/memory.json')
        print(f"Processed {len(messages)} messages and saved to model/memory.json")
        
        # Add timing information
        start_time = time.time()
        print("Chunking messages into Q/A format (this may take a few minutes)...")
        
        # Ask user for chunk limit
        limit_input = input("Enter maximum number of chunks to process (or press Enter for all): ").strip()
        chunk_limit = int(limit_input) if limit_input.isdigit() and int(limit_input) > 0 else None
        
        # Process chunks with optional limit
        model_name = 'default'  # You may want to prompt or infer this
        chunks = chunk_messages_into_qa(model_name=model_name, chunk_size=2000, limit=chunk_limit)
        
        elapsed_time = time.time() - start_time
        print(f"Memory data chunked into {len(chunks)} Q/A segments in {elapsed_time:.2f} seconds")
        
        # Now proceed with creating conversation memory
        print("Creating conversation memory and summaries...")
        start_time = time.time()
        
        # Initialize LLM once
        llm = initialize_llm()
        
        # Create conversation memory
        conversation_memory = create_conversational_memory(chunks, llm, model_name=model_name)
        
        # Save the conversation chain if needed
        if conversation_memory:
            # Save some test memory for verification
            test_response = conversation_memory({"question": "Tell me about our conversations"})
            with open('model/memory_test.json', 'w', encoding='utf-8') as f:
                json.dump(test_response, f, default=str, ensure_ascii=False, indent=2)
            
            print("Conversation memory created successfully!")
        
        elapsed_time = time.time() - start_time
        print(f"Memory processing completed in {elapsed_time:.2f} seconds")
        
        print("Setup complete! You can now have natural conversations with your AI friend.")

        if conversation_memory:
            # Save some test memory for verification
            test_response = conversation_memory({"question": "Tell me about our conversations"})
            with open('model/memory_test.json', 'w', encoding='utf-8') as f:
                json.dump(test_response, f, default=str, ensure_ascii=False, indent=2)
            
            # Save chat history in JSONL format for future loading
            save_chat_history_to_jsonl(conversation_memory)
            
            print("Conversation memory and chat history created successfully!")