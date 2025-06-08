import json
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import emoji
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import platform

# --- Configuration ---
DEFAULT_MODEL_DIR = 'model//Defautl'
ANALYSIS_RESULTS_DIR = 'analysis_results'
MAX_MESSAGES = 100000  # Maximum messages to analyze
# Set appropriate number of workers based on OS
if platform.system() == 'Windows':
    NUM_WORKERS = min(61, multiprocessing.cpu_count())  # Windows process limit
else:
    NUM_WORKERS = multiprocessing.cpu_count()
SKIP_VISUALIZATIONS = True  # Skip most visualizations for speed
CHUNK_SIZE = 1000  # Process messages in chunks

# --- Data Loading and Preprocessing ---
def load_chat_data(file_path=None):
    """Load chat data from JSON file"""
    if file_path is None:
        file_path = os.path.join(DEFAULT_MODEL_DIR, 'memory.json')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        participants = data.get('participants', [])
        messages = data.get('messages', [])
        
        print(f"Loaded chat data with {len(messages)} messages from {len(participants)} participants")
        print(f"Using {NUM_WORKERS} workers for processing")
        
        if len(messages) > MAX_MESSAGES:
            print(f"Limiting analysis to {MAX_MESSAGES} most recent messages")
            messages = messages[-MAX_MESSAGES:]
            data['messages'] = messages
            
        return data
    except Exception as e:
        print(f"Error loading chat data: {str(e)}")
        return None

def process_message_batch(messages_batch):
    """Process a batch of messages"""
    processed_data = []
    for msg in messages_batch:
        if not isinstance(msg, dict):
            continue
            
        # Extract sender name
        sender = msg.get('sender_name', '')
        if not sender and 'sender' in msg:
            sender = msg['sender']
        if not sender:
            sender = 'Unknown'
        
        # Extract content
        content = msg.get('content', '')
        if not content or content.startswith('https://') or len(content) < 2:
            continue
            
        # Skip system messages
        if 'แสดงความรู้สึก' in content or 'ต่อข้อความของคุณ' in content:
            continue
        
        # Process the message
        processed_msg = {
            'sender': sender,
            'content': content,
            'timestamp': msg.get('timestamp', ''),
            'type': 'text',
            'has_emoji': bool(emoji.emoji_count(content)),
            'char_count': len(content),
            'word_count': len(content.split()),  # Simplified word count
            'sentence_count': content.count('.') + content.count('!') + content.count('?') + 1
        }
        
        # Determine message type
        if '?' in content or content.endswith(('?', '??', '???')):
            processed_msg['type'] = 'question'
        elif any(q in content.lower() for q in ['ทำไม', 'อะไร', 'ที่ไหน', 'เมื่อไหร่', 'ใคร', 'อย่างไร']):
            processed_msg['type'] = 'question'
        elif len(content) < 15:
            processed_msg['type'] = 'short_response'
        else:
            processed_msg['type'] = 'statement'
        
        processed_data.append(processed_msg)
    
    return processed_data

def preprocess_messages(data):
    """Extract and preprocess message content using parallel processing"""
    messages = data.get('messages', [])
    print(f"Preprocessing {len(messages)} messages...")
    
    # Split messages into chunks
    chunks = [messages[i:i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]
    print(f"Split into {len(chunks)} chunks of size {CHUNK_SIZE}")
    
    processed_data = []
    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(process_message_batch, chunk) for chunk in chunks]
            
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Processing chunks"):
                result = future.result()
                if result:
                    processed_data.extend(result)
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        print("Falling back to sequential processing...")
        # Fallback to sequential processing if parallel processing fails
        for chunk in tqdm(chunks, desc="Processing chunks sequentially"):
            result = process_message_batch(chunk)
            if result:
                processed_data.extend(result)
    
    print(f"Preprocessing complete! Processed {len(processed_data)} valid messages")
    return processed_data

# --- Optimized Analysis Functions ---
def analyze_message_distribution(processed_data):
    """Analyze message distribution by sender and type"""
    print("Analyzing message distribution...")
    df = pd.DataFrame(processed_data)
    os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
    
    # Calculate all statistics in one pass
    stats = {
        'total_messages': len(df),
        'unique_senders': df['sender'].nunique(),
        'message_types': df['type'].value_counts().to_dict(),
        'avg_message_length': df['char_count'].mean(),
        'avg_word_count': df['word_count'].mean(),
        'senders': df['sender'].value_counts().to_dict(),
        'emoji_usage': {
            'total_with_emoji': df['has_emoji'].sum(),
            'percentage_with_emoji': (df['has_emoji'].mean() * 100)
        },
        'message_length_stats': {
            'min': df['char_count'].min(),
            'max': df['char_count'].max(),
            'mean': df['char_count'].mean(),
            'median': df['char_count'].median()
        }
    }
    
    # Save statistics
    with open(f'{ANALYSIS_RESULTS_DIR}/basic_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return df

def analyze_message_length(df):
    """Analyze message length metrics efficiently"""
    print("Analyzing message lengths...")
    
    # Calculate all length metrics in one pass
    length_stats = df.groupby('sender').agg({
        'char_count': ['mean', 'median', 'max', 'min', 'std'],
        'word_count': ['mean', 'median', 'max'],
        'sentence_count': ['mean', 'median', 'max']
    }).round(2)
    
    # Save to CSV
    length_stats.to_csv(f'{ANALYSIS_RESULTS_DIR}/message_length_summary.csv')
    
    return length_stats

def analyze_word_usage(df):
    """Analyze word usage patterns efficiently"""
    print("Analyzing word usage...")
    
    # Sample data if too large
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
    
    # Analyze emoji usage efficiently
    emoji_counts = Counter()
    for content in df['content']:
        emojis = [c for c in content if emoji.is_emoji(c)]
        emoji_counts.update(emojis)
    
    # Get top emojis
    top_emojis = dict(emoji_counts.most_common(10))
    
    # Save emoji stats
    emoji_stats = {
        'total_emojis': sum(emoji_counts.values()),
        'unique_emojis': len(emoji_counts),
        'top_emojis': top_emojis
    }
    
    with open(f'{ANALYSIS_RESULTS_DIR}/emoji_stats.json', 'w', encoding='utf-8') as f:
        json.dump(emoji_stats, f, ensure_ascii=False, indent=2)
    
    return emoji_counts

def analyze_question_patterns(df):
    """Analyze question patterns efficiently"""
    print("Analyzing question patterns...")
    
    # Filter questions once
    questions_df = df[df['type'] == 'question'].copy()
    
    if len(questions_df) > 500:
        questions_df = questions_df.sample(n=500, random_state=42)
    
    # Define question type mapping
    question_types = {
        'why': ['ทำไม'],
        'what': ['อะไร'],
        'where': ['ที่ไหน'],
        'when': ['เมื่อไหร่', 'เมื่อไร'],
        'who': ['ใคร'],
        'how': ['อย่างไร', 'ยังไง']
    }
    
    # Process questions efficiently
    def get_question_type(question):
        question = question.lower()
        if question.endswith(('?', '??', '???')):
            return 'yes/no'
        for q_type, keywords in question_types.items():
            if any(keyword in question for keyword in keywords):
                return q_type
        return 'other'
    
    # Apply question type analysis
    questions_df['question_type'] = questions_df['content'].apply(get_question_type)
    
    # Calculate question statistics
    question_stats = {
        'total_questions': len(questions_df),
        'question_types': questions_df['question_type'].value_counts().to_dict(),
        'common_questions': questions_df['content'].value_counts().head(20).to_dict()
    }
    
    # Save question stats
    with open(f'{ANALYSIS_RESULTS_DIR}/question_stats.json', 'w', encoding='utf-8') as f:
        json.dump(question_stats, f, ensure_ascii=False, indent=2)
    
    return questions_df

def create_summary_report(df, length_summary, emoji_counts):
    """Create an efficient summary report"""
    print("Creating summary report...")
    
    # Calculate all statistics in one pass
    stats = {
        'overview': {
            'total_messages': len(df),
            'unique_senders': df['sender'].nunique(),
            'messages_with_emoji': df['has_emoji'].sum(),
            'emoji_percentage': df['has_emoji'].mean() * 100,
            'total_questions': len(df[df['type'] == 'question'])
        },
        'message_distribution': df['sender'].value_counts().to_dict(),
        'message_types': df['type'].value_counts().to_dict(),
        'length_stats': {
            sender: {
                'avg_chars': group['char_count'].mean(),
                'avg_words': group['word_count'].mean(),
                'max_chars': group['char_count'].max()
            }
            for sender, group in df.groupby('sender')
        },
        'top_emojis': dict(sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    }
    
    # Save comprehensive stats
    with open(f'{ANALYSIS_RESULTS_DIR}/comprehensive_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Create markdown report
    with open(f'{ANALYSIS_RESULTS_DIR}/chat_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# Chat Dialogue Analysis Report\n\n")
        
        # Overview section
        f.write("## Overview\n")
        f.write(f"- Total messages analyzed: {stats['overview']['total_messages']}\n")
        f.write(f"- Unique senders: {stats['overview']['unique_senders']}\n")
        f.write(f"- Messages with emojis: {stats['overview']['messages_with_emoji']} ({stats['overview']['emoji_percentage']:.1f}%)\n")
        f.write(f"- Questions asked: {stats['overview']['total_questions']}\n\n")
        
        # Message distribution
        f.write("## Message Distribution\n")
        for sender, count in stats['message_distribution'].items():
            f.write(f"- {sender}: {count} messages\n")
        f.write("\n")
        
        # Message types
        f.write("## Message Types\n")
        for type_name, count in stats['message_types'].items():
            f.write(f"- {type_name}: {count} ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        # Length statistics
        f.write("## Message Length Analysis\n")
        f.write("| Sender | Avg Chars | Avg Words | Max Chars |\n")
        f.write("|--------|-----------|-----------|----------|\n")
        for sender, metrics in stats['length_stats'].items():
            f.write(f"| {sender} | {metrics['avg_chars']:.1f} | {metrics['avg_words']:.1f} | {metrics['max_chars']} |\n")
        f.write("\n")
        
        # Emoji statistics
        if stats['top_emojis']:
            f.write("## Top Emojis\n")
            f.write("| Emoji | Count |\n")
            f.write("|-------|-------|\n")
            for emoji, count in stats['top_emojis'].items():
                f.write(f"| {emoji} | {count} |\n")
            f.write("\n")

# --- Main Function ---
def main():
    """Main function to run the analysis"""
    start_time = datetime.now()
    print(f"Starting chat dialogue analysis at {start_time}")
    
    data = load_chat_data()
    if not data:
        print("Failed to load data. Exiting.")
        return
    
    processed_data = preprocess_messages(data)
    if not processed_data:
        print("No valid messages to analyze. Exiting.")
        return
    
    df = analyze_message_distribution(processed_data)
    
    try:
        length_summary = analyze_message_length(df)
    except Exception as e:
        print(f"Error analyzing message length: {str(e)}")
        length_summary = pd.DataFrame()
    
    try:
        emoji_counts = analyze_word_usage(df)
    except Exception as e:
        print(f"Error analyzing word usage: {str(e)}")
        emoji_counts = {}
    
    try:
        analyze_question_patterns(df)
    except Exception as e:
        print(f"Error analyzing question patterns: {str(e)}")
    
    try:
        create_summary_report(df, length_summary, emoji_counts)
    except Exception as e:
        print(f"Error creating summary report: {str(e)}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Analysis complete! Results saved to the '{ANALYSIS_RESULTS_DIR}' directory.")
    print(f"Total analysis time: {duration}")

if __name__ == "__main__":
    main()
