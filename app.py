import os
import tempfile
import json
import threading
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import csv

# LangChain imports - updated for compatibility
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Local imports
import modeling
from modeling import process_participant_model, initialize_llm, create_conversational_memory, chunk_messages_into_qa
from chat_rag import build_and_save_index, get_retriever, get_fewshot_qa_context

# Load environment variables
load_dotenv()

# --- Configuration ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.secret_key = 'your_secret_key_here'

# Global variables
processing_status = {}
llm_instances = {}  # Cache LLM instances per model
conversation_histories = {}  # Store conversation histories per model
model_chat_histories = {}  # Store chat histories per model

# --- Utility Functions ---
def ensure_model_structure():
    """Ensure that the basic model directories and files exist."""
    os.makedirs('model', exist_ok=True)
    os.makedirs('model/default', exist_ok=True)
    
    # Create default files if they don't exist
    default_files = {
        'model/default/memory_summaries.json': [],
        'model/default/memory.json': {"participants": [], "messages": []},
        'model/models.json': ['default']
    }
    
    for file_path, default_content in default_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_content, f)
    
    os.makedirs('model/default/chunks', exist_ok=True)

def load_memory_summaries(model_name='default'):
    """Load memory summaries for a specific model."""
    try:
        summary_path = f'model/{model_name}/memory_summaries.json'
        fallback_path = 'model/memory_summaries.json'
        
        # Try model-specific path first, then fallback
        if not os.path.exists(summary_path):
            if not os.path.exists(fallback_path):
                os.makedirs(f'model/{model_name}', exist_ok=True)
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
                return []
            else:
                summary_path = fallback_path
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle both formats - new format with metadata and summaries, or old format
        return data.get("summaries", data) if isinstance(data, dict) else data
        
    except Exception as e:
        print(f"Error loading memory summaries for {model_name}: {e}")
        try:
            os.makedirs(f'model/{model_name}', exist_ok=True)
            with open(f'model/{model_name}/memory_summaries.json', 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as create_error:
            print(f"Error creating empty memory_summaries.json: {create_error}")
        return []

def extract_chat_style(memory_summaries):
    """Extract chat style information from memory summaries."""
    chat_style = {
        "names": [],
        "particles": [],
        "topics": [],
        "tone": "neutral"
    }
    
    if not memory_summaries:
        return chat_style
    
    all_text = "".join(batch.get("summary", "") + "\n\n" for batch in memory_summaries)
    
    # Extract names - enhanced patterns for various formats
    name_patterns = [
        r'(?:ชื่อ|เรียก|คือ)[^,\.]*?["\']([^"\']{2,20})["\']',
        r'([a-zA-Z0-9\u0E00-\u0E7F\._]{2,15})\s*(?:และ|กับ)\s*([a-zA-Z0-9\u0E00-\u0E7F\._]{2,15})',
        r'คุณ\s*([a-zA-Z0-9\u0E00-\u0E7F\._]{2,15})',
        r'([a-zA-Z0-9\u0E00-\u0E7F\._]{2,15})\s*พูด',
        r'@([a-zA-Z0-9\u0E00-\u0E7F\._]{2,15})',
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, all_text)
        for match in matches:
            if isinstance(match, str) and len(match.strip()) > 1:
                name = match.strip()
                if name not in chat_style["names"] and not name.isdigit():
                    chat_style["names"].append(name)
            elif isinstance(match, tuple):
                for name in match:
                    name = name.strip()
                    if len(name) > 1 and name not in chat_style["names"] and not name.isdigit():
                        chat_style["names"].append(name)
    
    # Extract particles - enhanced patterns for various ending particles
    particle_patterns = [
        r'["\']([ค้าบ|จ้า|นะ|ย|เลย|อ่ะ|ว่ะ|จริง|มั้ง|ไง|หว่า|ครับ|ค่ะ|จ๊ะ|แล้ว][^"\']*?)["\']',
        r'มักใช้\s*["\']([^"\']{1,6})["\']',
    ]
    
    for pattern in particle_patterns:
        matches = re.findall(pattern, all_text)
        for match in matches:
            particle = match.strip()
            if len(particle) <= 6 and particle not in chat_style["particles"]:
                chat_style["particles"].append(particle)
    
    # Extract topics
    topic_keywords = [
        "Minecraft", "เกม", "game", "โครงงาน", "โปรเจกต์", "project", 
        "นำเสนอ", "เรียน", "สอบ", "ท่องเที่ยว", "ทริป", "อาหาร", 
        "กีฬา", "ภาพยนตร์", "หนัง", "เพลง", "music", "งาน", "work",
        "ครอบครัว", "family", "เพื่อน", "friend", "ปาร์ตี้", "party"
    ]
    
    chat_style["topics"] = [keyword for keyword in topic_keywords 
                           if keyword.lower() in all_text.lower()]
    
    # Enhanced tone detection
    tone_indicators = {
        "very_friendly": ["สนิทสนม", "เป็นกันเองมาก", "ตลก", "ขี้เล่น", "แซว", "ล้อเล่น", "สนุก", "ฮา"],
        "friendly": ["เป็นกันเอง", "ดี", "น่ารัก", "cute", "สบาย", "ผ่อนคลาย"],
        "formal": ["ทางการ", "สุภาพ", "เป็นทางการ", "เรียบร้อย", "formal", "official"],
        "excited": ["ตื่นเต้น", "excited", "เฮ้ย", "ว้าว", "เจ๋ง", "เท่", "cool"],
        "casual": ["ชิล", "chill", "ปกติ", "ธรรมดา", "normal", "ok"]
    }
    
    tone_scores = {tone: 0 for tone in tone_indicators.keys()}
    
    for tone, indicators in tone_indicators.items():
        for indicator in indicators:
            tone_scores[tone] += all_text.lower().count(indicator.lower())
    
    if max(tone_scores.values()) > 0:
        chat_style["tone"] = max(tone_scores, key=tone_scores.get)
    
    return chat_style

def get_or_create_llm(model_name):
    """Get or create LLM instance for a specific model."""
    if model_name not in llm_instances:
        try:
            print(f"Initializing LLM for model: {model_name}")
            llm_instances[model_name] = initialize_llm()
        except Exception as e:
            print(f"Error initializing LLM for {model_name}: {e}")
            return None
    return llm_instances[model_name]

def create_system_message(chat_style, memory_context, few_shot_context=""):
    """Create a system message based on chat style and context."""
    names_str = "และ ".join(chat_style["names"]) if chat_style["names"] else "เพื่อน"
    particles_str = ", ".join(f"'{p}'" for p in chat_style["particles"]) if chat_style["particles"] else "'นะ', 'ครับ', 'ค่ะ'"
    topics_str = ", ".join(chat_style["topics"]) if chat_style["topics"] else "เรื่องทั่วไป"
    
    tone_desc = "เพื่อนสนิทมาก" if chat_style["tone"] == "very_friendly" else "เพื่อน"
    banter_note = "มีการแซวเล่นกันบ้างตามโอกาสที่เหมาะสม" if chat_style["tone"] == "very_friendly" else ""
    
    system_msg = f"""คุณคือเพื่อนสนิทที่ชื่อ "{names_str.split()[0] if ' ' in names_str else names_str}" ในบทสนทนากับเพื่อนของคุณ

ข้อมูลเกี่ยวกับความสัมพันธ์ของคุณกับเพื่อน:
{memory_context}

{f"ตัวอย่างการพูดคุยกับเพื่อนของคุณ:{few_shot_context}" if few_shot_context else ""}

พูดคุยอย่างเป็นธรรมชาติแบบ{tone_desc} 
ใช้คำเรียกอย่างเป็นกันเอง ใช้คำลงท้ายแบบสนิทสนม เช่น {particles_str}
{banter_note}
{"แบ่งปันประสบการณ์ร่วมกันเกี่ยวกับ " + topics_str if topics_str else ""}

พยายามเลียนแบบรูปแบบการสนทนาที่ปรากฏในความทรงจำและตัวอย่างของคุณ
ใช้ศัพท์แสลง คำลงท้าย และสำนวนที่ปรากฏในตัวอย่างเพื่อให้การสนทนาเป็นธรรมชาติที่สุด
ห้ามเอ่ยถึงว่าคุณกำลังดูข้อมูลจากความทรงจำหรือการสนทนาที่ผ่านมา ให้ตอบโดยเป็นธรรมชาติเหมือนเพื่อนที่กำลังคุยกันสดๆ 
และอย่าพูดว่าคุณเป็น AI หรือโมเดลภาษา"""
    
    return system_msg

def create_conversation_chain(llm, retriever, chat_style, memory_context, few_shot_context, chat_history):
    """Create a conversation chain with the given components."""
    names_str = "และ ".join(chat_style["names"]) if chat_style["names"] else "เพื่อน"
    particles_str = ", ".join(f"'{p}'" for p in chat_style["particles"]) if chat_style["particles"] else "'นะ', 'ครับ', 'ค่ะ'"
    topics_str = ", ".join(chat_style["topics"]) if chat_style["topics"] else "เรื่องทั่วไป"
    
    if retriever:
        # Create a safer retriever wrapper to avoid attribute errors
        def safe_retrieve(query):
            try:
                if hasattr(retriever, 'get_relevant_documents'):
                    return retriever.get_relevant_documents(query)
                elif hasattr(retriever, 'invoke'):
                    return retriever.invoke(query)
                else:
                    # Fallback to direct retrieval
                    return retriever.similarity_search(query, k=4)
            except Exception as e:
                print(f"Error in retrieval: {e}")
                return []
        
        # System prompt for retrieval-based conversation - fixed formatting
        system_prompt_template = """คุณชื่อ "{friend_name}" และกำลังสนทนากับเพื่อนของคุณ

ใช้ข้อมูลจากความทรงจำและประสบการณ์ที่ผ่านมาเหล่านี้ในการตอบ:
{memory_context}

ตัวอย่างการพูดคุยที่ผ่านมา:
{few_shot_context}

ให้ตอบด้วยภาษาไทยที่{tone_style} 
มีการใช้คำลงท้ายอย่าง {particles_str}
{banter_note}

เลียนแบบวิธีการพูดคุย คำศัพท์ และลักษณะการสื่อสารที่ปรากฏในข้อมูลข้างต้น
ห้ามระบุว่ากำลังใช้ข้อมูลจากความทรงจำหรือบอกว่าเป็น AI ให้ตอบเหมือนคุณเป็นเพื่อนจริงๆ

บริบทที่เกี่ยวข้อง: {{context}}

ประวัติการสนทนา:
{{chat_history}}

คำถาม: {{question}}

ตอบ:"""

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def get_context_and_history(inputs):
            question = inputs["question"]
            # Get relevant documents
            docs = safe_retrieve(question)
            context = format_docs(docs)
            
            # Format chat history
            history_text = ""
            if chat_history:
                for msg in chat_history[-6:]:  # Last 6 messages for context
                    if isinstance(msg, HumanMessage):
                        history_text += f"เพื่อน: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"{names_str.split()[0] if ' ' in names_str else names_str}: {msg.content}\n"
            
            return {
                "context": context,
                "question": question,
                "chat_history": history_text
            }

        # Format the system prompt with known variables
        formatted_system_prompt = system_prompt_template.format(
            friend_name=names_str.split()[0] if ' ' in names_str else names_str,
            memory_context=memory_context,
            few_shot_context=few_shot_context,
            tone_style='เป็นกันเองมาก สนิทสนม' if chat_style["tone"] == "very_friendly" else 'เป็นกันเอง',
            particles_str=particles_str,
            banter_note='มีการแซวเล่นตามโอกาสที่เหมาะสม' if chat_style["tone"] == "very_friendly" else '',
            #topics_note='พูดคุยเกี่ยวกับ ' + topics_str if topics_str else ''
        )

        # Create the chain
        chain = (
            RunnableLambda(get_context_and_history) |
            RunnableLambda(lambda x: formatted_system_prompt.format(**x)) |
            llm |
            StrOutputParser()
        )
        
        return chain
    else:
        # Simple conversation without retrieval - fixed formatting
        system_prompt_template = """คุณคือเพื่อนสนิทชื่อ "{friend_name}" และกำลังสนทนากับเพื่อนของคุณ

ใช้ความทรงจำเกี่ยวกับมิตรภาพของคุณในการตอบ:
{memory_context}

ตัวอย่างการพูดคุยที่ผ่านมา:
{few_shot_context}

ให้ตอบด้วยภาษาไทยที่{tone_style}
มีการใช้คำลงท้ายอย่าง {particles_str}
{banter_note}

**Dont use word เครๆ**
พยายามเลียนแบบวิธีการพูดคุย คำศัพท์ และลักษณะการสื่อสารที่ปรากฏในข้อมูลข้างต้น
ห้ามระบุว่ากำลังใช้ข้อมูลจากความทรงจำหรือบอกว่าเป็น AI ให้ตอบเหมือนคุณเป็นเพื่อนจริงๆ

ประวัติการสนทนา:
{{chat_history}}

คำถาม: {{question}}

ตอบ:"""

        def get_history_and_question(inputs):
            question = inputs["question"]
            
            # Format chat history
            history_text = ""
            if chat_history:
                for msg in chat_history[-6:]:  # Last 6 messages for context
                    if isinstance(msg, HumanMessage):
                        history_text += f"เพื่อน: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        history_text += f"{names_str.split()[0] if ' ' in names_str else names_str}: {msg.content}\n"
            
            return {
                "question": question,
                "chat_history": history_text
            }

        # Format the system prompt with known variables
        formatted_system_prompt = system_prompt_template.format(
            friend_name=names_str.split()[0] if ' ' in names_str else names_str,
            memory_context=memory_context,
            few_shot_context=few_shot_context,
            tone_style='เป็นกันเองมาก สนิทสนม' if chat_style["tone"] == "very_friendly" else 'เป็นกันเอง',
            particles_str=particles_str,
            banter_note='มีการแซวเล่นตามโอกาสที่เหมาะสม ใช้อีโมจิบ้างเพื่อแสดงอารมณ์' if chat_style["tone"] == "very_friendly" else '',
            #topics_note='อ้างอิงถึงเรื่องราวที่เคยคุยกันมาก่อน เช่น ' + topics_str if topics_str else ''
        )
        print(f"Formatted system prompt: {formatted_system_prompt}")
        chain = (
            RunnableLambda(get_history_and_question) |
            RunnableLambda(lambda x: formatted_system_prompt.format(**x)) |
            llm |
            StrOutputParser()
        )
        
        return chain

def invoke_conversation_chain(chain, user_message):
    """Invoke the conversation chain and extract the response properly."""
    try:
        result = chain.invoke({"question": user_message})
        
        # Handle different response types
        if isinstance(result, str):
            return result
        elif hasattr(result, 'content'):
            return result.content
        elif isinstance(result, dict):
            # Handle different response formats
            for key in ["answer", "text", "result", "output"]:
                if key in result:
                    return str(result[key])
            return str(result)
        else:
            return str(result)
            
    except Exception as e:
        print(f"Error in conversation chain: {e}")
        import traceback
        traceback.print_exc()
        return f"ขอโทษนะ เกิดข้อผิดพลาดขึ้น: {str(e)}"

# Initialize model structure
ensure_model_structure()

# --- Routes ---
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/direct_chat')
def direct_chat():
    return render_template('direct_chat.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/setup')
def setup():
    return render_template('setup.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['chatfile']
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    # Parse participants
    with open(temp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    participants = data.get('participants', [])
    
    # Ensure participants are strings
    processed_participants = []
    for p in participants:
        if isinstance(p, dict) and 'name' in p:
            processed_participants.append(p['name'])
        elif isinstance(p, str):
            processed_participants.append(p)
        else:
            processed_participants.append(str(p))
    
    session['uploaded_file'] = temp_path
    return jsonify({'participants': processed_participants})

@app.route('/select_participant', methods=['POST'])
def select_participant():
    req = request.get_json()
    participant = req['participant']
    model_name = req['model_name']
    chunk_size = req.get('chunk_size', 5)
    
    # Save to session
    session.update({
        'ai_participant': participant,
        'model_name': model_name,
        'chunk_size': chunk_size
    })
    
    uploaded_file = session.get('uploaded_file')
    if not uploaded_file:
        return jsonify({'status': 'error', 'msg': 'No file uploaded'}), 400
    
    # Process the uploaded file
    messages = modeling.load_and_process_messages(uploaded_file)
    
    # Create model-specific directory structure
    model_dir = f'model/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f'{model_dir}/chunks', exist_ok=True)
    
    # Move memory.json to the model-specific directory
    if os.path.exists('memory.json'):
        os.replace('memory.json', f'{model_dir}/memory.json')
    
    # Create empty memory_summaries.json if it doesn't exist
    if not os.path.exists(f'{model_dir}/memory_summaries.json'):
        with open(f'{model_dir}/memory_summaries.json', 'w') as f:
            json.dump([], f)
    
    # Create FAISS index
    build_and_save_index(model_name)
    
    # Initialize LLM for this model
    get_or_create_llm(model_name)

    # --- NEW: Chunk and summarize memory for buffer memory ---
    # Use a reasonable chunk size for initial summarization
    chunks = chunk_messages_into_qa(model_name=model_name, chunk_size=2000)
    llm = get_or_create_llm(model_name)
    
    # Create participant info file
    participant_info = {
        'ai_participant': participant,
        'other_participants': [],
        'created_at': datetime.now().isoformat()
    }
    
    # Load participants from memory.json
    try:
        with open(f'{model_dir}/memory.json', 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
            all_participants = memory_data.get('participants', [])
            # Store other participants (excluding the AI participant)
            participant_info['other_participants'] = [p for p in all_participants if p != participant]
    except Exception as e:
        print(f"Error loading participants: {e}")
    
    # Save participant info
    with open(f'{model_dir}/participant_info.json', 'w', encoding='utf-8') as f:
        json.dump(participant_info, f, ensure_ascii=False, indent=2)
    
    # This will create memory_summaries.json and memory_vectorstore for this model
    create_conversational_memory(chunks, llm, model_name=model_name)
    
    # Store processing status
    session['processing_status'] = {
        'status': 'started',
        'model_name': model_name,
        'participant': participant,
        'start_time': datetime.now().isoformat()
    }
    
    return jsonify({
        'status': 'processing', 
        'message': f'Model "{model_name}" created successfully. You can now start chatting!',
        'participant_info': participant_info
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    models = []
    current_model = session.get('model_name')
    if current_model:
        models.append(current_model)
    
    try:
        # Check models.json for backward compatibility
        if os.path.exists('model/models.json'):
            with open('model/models.json', 'r', encoding='utf-8') as f:
                stored_models = json.load(f)
                models.extend(model for model in stored_models if model not in models)
        
        # Scan model directory
        for item in os.listdir('model'):
            dir_path = os.path.join('model', item)
            if (os.path.isdir(dir_path) and 
                os.path.exists(os.path.join(dir_path, 'memory.json')) and 
                item not in models and item != 'chunks'):
                models.append(item)
                
    except Exception as e:
        print(f"Error scanning for models: {e}")
    
    return jsonify({'models': models})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        selected_model = request.json.get('model', '') or session.get('model_name', 'default')
        
        if not user_message.strip():
            return jsonify({'error': 'Please enter a message'}), 400
        
        session['model_name'] = selected_model
        print(f"Processing chat with model: {selected_model}")
        
        # Get or initialize LLM for this model
        llm = get_or_create_llm(selected_model)
        if not llm:
            return jsonify({'error': 'Failed to initialize LLM for this model'}), 500
        
        # Get or create conversation history for this model
        if selected_model not in model_chat_histories:
            model_chat_histories[selected_model] = []
        
        chat_history = model_chat_histories[selected_model]
        
        # Load model-specific data
        retriever = None
        try:
            retriever = get_retriever(selected_model)
        except Exception as e:
            print(f"Warning: Could not get retriever for {selected_model}: {e}")
        
        # Load participant info
        participant_info = {}
        try:
            with open(f'model/{selected_model}/participant_info.json', 'r', encoding='utf-8') as f:
                participant_info = json.load(f)
        except Exception as e:
            print(f"Error loading participant info: {e}")
        
        memory_summaries = load_memory_summaries(selected_model)
        chat_style = extract_chat_style(memory_summaries)
        
        # Override names from participant info if available
        if participant_info:
            chat_style["names"] = [participant_info['ai_participant']]
            if 'other_participants' in participant_info:
                chat_style["other_participants"] = participant_info['other_participants']
        
        # If no names extracted from summaries or participant info, check memory.json
        if not chat_style["names"]:
            try:
                memory_path = f'model/{selected_model}/memory.json'
                if os.path.exists(memory_path):
                    with open(memory_path, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                        if 'participants' in memory_data and memory_data['participants']:
                            chat_style["names"] = memory_data['participants']
            except Exception as e:
                print(f"Error loading participant names from memory.json: {e}")
        
        # Build memory context
        memory_context = "".join(batch.get("summary", "") + "\n\n" for batch in memory_summaries)
        
        # Add participant information to memory context
        if participant_info:
            memory_context = f"""ข้อมูลผู้สนทนา:
- คุณคือ: {participant_info['ai_participant']}
- คู่สนทนา: {', '.join(participant_info['other_participants'])}

{memory_context}"""
        
        # Inject summarized memory into buffer memory for LLM
        if memory_summaries:
            summary_text = "\n\n".join(batch.get("summary", "") for batch in memory_summaries if batch.get("summary"))
            if summary_text:
                # Only add if not already present
                if not chat_history or not (isinstance(chat_history[0], SystemMessage) and summary_text in chat_history[0].content):
                    chat_history.insert(0, SystemMessage(content=summary_text))
        
        # Get few-shot examples
        few_shot_context = ""
        try:
            few_shot_context = get_fewshot_qa_context(user_message, top_k=3, format_style='qa', model_name=selected_model)
            print(f"Retrieved {few_shot_context.count('(Rag)')} few-shot examples")
        except Exception as e:
            print(f"Error getting few-shot examples: {e}")
        
        # Build conversation chain
        langchain_conversation = create_conversation_chain(
            llm, retriever, chat_style, memory_context, few_shot_context, chat_history
        )
        
        # Get baseline response
        names_str = "และ ".join(chat_style["names"]) if chat_style["names"] else "เพื่อน"
        particles_str = ", ".join(f"'{p}'" for p in chat_style["particles"]) if chat_style["particles"] else "'นะ', 'ครับ', 'ค่ะ'"
        topics_str = ", ".join(chat_style["topics"]) if chat_style["topics"] else "เรื่องทั่วไป"
        
        
        baseline_prompt = """
เราคือเพื่อนสนิทที่คุยกันแบบสบาย ๆ เป็นกันเอง ไม่ต้องเป็นทางการ พูดให้เข้าใจง่าย กระชับ ไม่ยืดเยื้อ เน้นความจริงใจ แต่ก็ยังสนุก มีความเป็นธรรมชาติ

สไตล์คำพูด:
    •    ใช้ "เราว่า…", "อันนี้น่าสนใจนะ", "ลองดูมั้ย", "เดี๋ยวส่งให้ดู"
    •    ชอบแทรกคำภาษาอังกฤษบ้าง เช่น "เทรนนี้โคตร cool", "เพิ่งไปเที่ยวมา"
    •    สื่อสารแบบเน้น essence ไม่อ้อมค้อม เช่น "อันนี้ตรง ๆ เลยคือ…", "พูดจริง ๆ นะ…"
    •    เวลาพูดเรื่องอะไรก็ตามจะไม่อวด แต่จะอธิบายแบบชวนเพื่อนคุย เช่น "มันเป็นแบบนี้นะ แต่น่าจะปรับได้อีก"

Tone: เป็นกันเอง มีพลัง ขี้เล่นแต่มี substance ไม่ใช้คำหยาบตรง ๆ แต่ถ้าอยู่กับเพื่อนสนิทอาจแอบมีคำแบบ "เห้ย", "โห" หรือ "แม่ง" บ้างในบางประโยคเพื่อเน้นอารมณ์

Example Conversation Opener:

"เห้ย เราเพิ่งไปเที่ยวมา แบบสนุกโคตร ๆ เดี๋ยวส่งรูปให้ดู"

หรือ

"เราว่าถ้าเอา idea นี้ไปทำจริง ๆ มีสิทธิ์ปังนะ ลองมาคุยกันมั้ย?

**Answer consise like talking with friend dont answer unnessary thing**
"""
        
        try:
            baseline_response = llm.invoke(baseline_prompt).content
        except Exception as e:
            print(f"Baseline model error: {str(e)}")
            baseline_response = f"Error with baseline model: {str(e)}"
        
        # Get LangChain response
        try:
            # Add user message to history
            #chat_history.append(HumanMessage(content=user_message))
            
            # Get response
            langchain_response = invoke_conversation_chain(langchain_conversation, user_message)
            
            # Add AI response to history
            #chat_history.append(AIMessage(content=langchain_response))
            
            # Keep only last 20 messages to prevent memory bloat
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                model_chat_histories[selected_model] = chat_history


                
        except Exception as e:
            import traceback
            print(f"Error with LangChain model: {str(e)}")
            traceback.print_exc()
            langchain_response = f"Error with LangChain model: {str(e)}"
        
        return jsonify({
            'baseline': baseline_response,
            'langchain': langchain_response,
            'model': selected_model,
            'status': 'success',
            'participant_info': participant_info
        })
    
    except Exception as e:
        import traceback
        print(f"Unexpected error in chat: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        winner = data.get('winner', '')
        user_message = data.get('message', '')
        baseline_response = data.get('baseline_response', '')
        langchain_response = data.get('langchain_response', '')
        model_name = data.get('model_name', '')
        eval_type = data.get('eval_type', 'natural_chat')
        evaluation_time = datetime.now().isoformat()
        
        # Create CSV file if it doesn't exist
        csv_path = 'Human_Eval.csv'
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    'timestamp',
                    'model_name',
                    'eval_type',
                    'user_message',
                    'baseline_response',
                    'langchain_response',
                    'winner',
                    'evaluation_time'
                ])
            
            # Write evaluation data
            writer.writerow([
                evaluation_time,
                model_name,
                eval_type,
                user_message,
                baseline_response,
                langchain_response,
                winner,
                evaluation_time
            ])
        
        return jsonify({
            'status': 'success',
            'message': 'Evaluation saved successfully'
        })
    except Exception as e:
        print(f"Error saving evaluation: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/eval_analysis', methods=['GET'])
def get_eval_analysis():
    try:
        csv_path = 'Human_Eval.csv'
        if not os.path.exists(csv_path):
            return jsonify({
                'total_evaluations': 0,
                'type_breakdown': {},
                'model_performance': {}
            })
        
        # Read the CSV file
        evaluations = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            evaluations = list(reader)
        
        # Calculate statistics
        total_evaluations = len(evaluations)
        
        # Type breakdown
        type_breakdown = {}
        for eval_type in ['natural_chat', 'memory_recall', 'personalization', 'emotional_support']:
            type_count = sum(1 for e in evaluations if e['eval_type'] == eval_type)
            type_breakdown[eval_type] = type_count
        
        # Model performance
        model_performance = {}
        for eval_type in ['natural_chat', 'memory_recall', 'personalization', 'emotional_support']:
            type_evals = [e for e in evaluations if e['eval_type'] == eval_type]
            if type_evals:
                baseline_wins = sum(1 for e in type_evals if e['winner'] == 'baseline')
                langchain_wins = sum(1 for e in type_evals if e['winner'] == 'langchain')
                total = len(type_evals)
                model_performance[eval_type] = {
                    'baseline_win_rate': round(baseline_wins / total * 100, 1),
                    'langchain_win_rate': round(langchain_wins / total * 100, 1),
                    'total_evaluations': total
                }
        
        return jsonify({
            'total_evaluations': total_evaluations,
            'type_breakdown': type_breakdown,
            'model_performance': model_performance
        })
    except Exception as e:
        print(f"Error getting evaluation analysis: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/update_models', methods=['POST'])
def update_models():
    req = request.get_json()
    models = req.get('models', [])
    
    try:
        os.makedirs('model', exist_ok=True)
        with open('model/models.json', 'w', encoding='utf-8') as f:
            json.dump(models, f)
        
        for model_name in models:
            os.makedirs(f'model/{model_name}', exist_ok=True)
            
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error saving models: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/process_model', methods=['POST'])
def process_model():
    """Start the LLM chunking and summarization process for a model"""
    req = request.get_json()
    model_name = req.get('model_name', session.get('model_name'))
    participant = req.get('participant', session.get('ai_participant'))
    chunk_size = req.get('chunk_size', session.get('chunk_size', 5))
    
    if not model_name:
        return jsonify({'status': 'error', 'message': 'No model name provided'}), 400
    
    model_dir = f'model/{model_name}'
    if not os.path.exists(model_dir) or not os.path.exists(f'{model_dir}/memory.json'):
        return jsonify({'status': 'error', 'message': 'Model does not exist or has no memory data'}), 404
    
    # Initialize progress tracking
    global processing_status
    processing_status[model_name] = {
        'status': 'started',
        'model_name': model_name,
        'participant': participant,
        'start_time': datetime.now().isoformat(),
        'progress': {
            'step': 1,
            'total_steps': 4,
            'current_step': 'chunking',
            'message': 'Starting the chunking process...',
            'chunks_processed': 0,
            'total_chunks': 0,
            'batches_processed': 0,
            'total_batches': 0
        }
    }
    
    def background_processing():
        try:
            result = process_participant_model(model_name, participant, chunk_size=chunk_size)
            
            if result.get('status') == 'success':
                processing_status[model_name]['status'] = 'completed'
                processing_status[model_name]['completed_time'] = datetime.now().isoformat()
                processing_status[model_name]['progress']['message'] = 'Processing completed successfully'
            else:
                processing_status[model_name]['status'] = 'error'
                processing_status[model_name]['error'] = result.get('message', 'Unknown error')
                processing_status[model_name]['progress']['message'] = f"Error: {result.get('message', 'Unknown error')}"
            
            processing_status[model_name]['result'] = result
        except Exception as e:
            import traceback
            traceback.print_exc()
            processing_status[model_name]['status'] = 'error'
            processing_status[model_name]['error'] = str(e)
            processing_status[model_name]['progress']['message'] = f"Error: {str(e)}"
    
    thread = threading.Thread(target=background_processing)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'processing',
        'message': 'Model processing started in the background',
        'model_name': model_name
    })

@app.route('/api/process_status', methods=['GET'])
def get_process_status():
    """Get the current processing status of models"""
    model_name = request.args.get('model_name', session.get('model_name'))
    
    if not model_name:
        return jsonify(processing_status)
    
    if model_name in processing_status:
        return jsonify(processing_status[model_name])
    else:
        return jsonify({'status': 'not_found', 'message': f'No processing data for model {model_name}'}), 404

# Error handlers for better UX
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Visit: http://localhost:5000")
    print("Setup page: http://localhost:5000/setup")
    print("Chat page: http://localhost:5000/index")
    app.run(debug=True, port=5000, host='0.0.0.0')
