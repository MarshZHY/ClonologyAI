from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os
from langchain.schema import Document 
from langchain_core.retrievers import BaseRetriever
from typing import List, Tuple

# Default model directory
DEFAULT_MODEL_DIR = 'model/default'

# Path templates for model files
def get_model_paths(model_name='default'):
    model_dir = f"model/{model_name}"
    return {
        'chat_history': f"{model_dir}/memory.json",
        'embeddings': f"{model_dir}/chat_embeddings.npy",
        'faiss_index': f"{model_dir}/chat_faiss_index.bin",
        'index_map': f"{model_dir}/chat_index.json"
    }

# Use default paths for backward compatibility
paths = get_model_paths()
CHAT_HISTORY_PATH = paths['chat_history']
EMBEDDINGS_PATH = paths['embeddings']
FAISS_INDEX_PATH = paths['faiss_index']
INDEX_MAP_PATH = paths['index_map']

# Load multilingual model (Thai support)
model = SentenceTransformer('BAAI/bge-m3')

def load_chat_texts(model_name='default'):
    """Load chat messages from memory.json and return as a list of texts."""
    paths = get_model_paths(model_name)
    chat_history_path = paths['chat_history']
    
    if not os.path.exists(chat_history_path):
        raise FileNotFoundError(f"{chat_history_path} not found.")
    with open(chat_history_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    messages = data.get('messages', [])
    texts = []
    for msg in messages:
        # You can adjust this to include sender, timestamp, etc.
        texts.append(msg.get('content', ''))
    return texts, messages

def build_and_save_index(model_name='default'):
    # Ensure model directory exists
    model_dir = f"model/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Get paths for this model
    paths = get_model_paths(model_name)
    
    texts, messages = load_chat_texts(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(paths['embeddings'], embeddings)
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, paths['faiss_index'])
    # Save index mapping (for retrieving message metadata)
    with open(paths['index_map'], 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"Chat embeddings and FAISS index saved for model: {model_name}")

def load_index(model_name='default'):
    paths = get_model_paths(model_name)
    embeddings = np.load(paths['embeddings'])
    index = faiss.read_index(paths['faiss_index'])
    with open(paths['index_map'], 'r', encoding='utf-8') as f:
        messages = json.load(f)
    return index, messages

def retrieve_similar_chats(query, top_k=5, model_name='default'):
    index, messages = load_index(model_name)
    query_emb = model.encode([query])
    distances, idxs = index.search(query_emb, top_k)
    results = []
    for i, idx in enumerate(idxs[0]):
        msg = messages[idx]
        results.append({
            'distance': float(distances[0][i]),
            'similarity': 1/(1+distances[0][i]),
            'message': msg
        })
    return results

def get_fewshot_qa_context(query, top_k=5, format_style='qa', model_name='default'):
    """
    Retrieve top-k similar user chat messages and their next replies as Q&A pairs for few-shot LLM context.
    format_style: 'qa' (default) or 'dialogue'.
    Returns a string to prepend to the LLM prompt.
    """
    index, messages = load_index(model_name)
    # Use all messages as possible Qs
    all_msgs = [(i, m) for i, m in enumerate(messages)]
    all_texts = [m['content'] for _, m in all_msgs]
    all_embs = model.encode(all_texts)
    # Encode query
    query_emb = model.encode([query])
    # Search
    import numpy as np
    from scipy.spatial.distance import cdist
    dists = cdist(query_emb, all_embs, metric='sqeuclidean')[0]
    top_indices = np.argsort(dists)[:top_k]
    # Build Q&A pairs with context (2 before, 2 after)
    qa_pairs = []
    context_window = 3
    for idx in top_indices:
        msg_idx, q_msg = all_msgs[idx]
        # Find the next message as the answer (if exists)
        if msg_idx+1 < len(messages):
            a_msg = messages[msg_idx+1]
            q = q_msg.get('content', '')
            a = a_msg.get('content', '')
            
            # Standardize sender names
            q_sender = "User"
            a_sender = "Assistant"

            # Gather context: 2 before and 2 after (excluding Q and A themselves)
            before_context = []
            for i in range(max(0, msg_idx-context_window), msg_idx):
                m = messages[i]
                # Standardize context sender names
                sender = "User" if m.get('sender', '').lower() != 'assistant' else "Assistant"
                before_context.append(f"{sender}: {m.get('content','')}")
            after_context = []
            for i in range(msg_idx+2, min(len(messages), msg_idx+2+context_window)):
                m = messages[i]
                # Standardize context sender names
                sender = "User" if m.get('sender', '').lower() != 'assistant' else "Assistant"
                after_context.append(f"{sender}: {m.get('content','')}")

            # Always use: sender: ...\nsender: ...\n for all context, Q, and A, no extra labels
            context_lines = []
            # Before context
            for line in before_context:
                context_lines.append(line)
            # Q and A, highlight the ragged text (the Q)
            # Highlight format: <RAG>user_1: ...</RAG> or similar
            if q.strip():
                context_lines.append(f"{q_sender} (Rag): {q}")
            else:
                context_lines.append(f"{q_sender}: {q}")
            context_lines.append(f"{a_sender}: {a}")
            # After context
            for line in after_context:
                context_lines.append(line)
            qa_pairs.append('\n'.join(context_lines))
    return '\n\n'.join(qa_pairs)

class FaissIndexRetriever(BaseRetriever):
    """A retriever that uses FAISS for semantic search and returns LangChain Documents."""
    
    def __init__(self, model_name='default'):
        self._model_name = model_name
        paths = get_model_paths(model_name)
        self._index_path = paths['faiss_index']
        self._map_path = paths['index_map']
        self._model = SentenceTransformer('BAAI/bge-m3')
        self._load_index()
        self.tags = []
    def _load_index(self):
        try:
            self._index = faiss.read_index(self._index_path)
            with open(self._map_path, 'r', encoding='utf-8') as f:
                self._messages = json.load(f)
            print(f"Successfully loaded index for model: {self._model_name}")
        except Exception as e:
            print(f"Error loading index for model {self._model_name}: {e}")
            # Initialize with empty data to avoid further errors
            self._index = None
            self._messages = []
            raise e
    
    def get_relevant_documents(self, query: str, top_k=3) -> List[Document]:
        """Retrieve documents relevant to the query."""
        if not self._index:
            print(f"No valid index loaded for model {self._model_name}")
            return []
            
        query_emb = self._model.encode([query])
        distances, indices = self._index.search(query_emb, top_k)
        
        docs = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._messages):  # Valid index check
                msg = self._messages[idx]
                content = msg.get('content', '')
                metadata = {
                    'distance': float(distances[0][i]),
                    'sender': msg.get('sender', ''),
                    'timestamp': msg.get('timestamp', ''),
                    'source': f"message_{idx}"
                }
                docs.append(Document(page_content=content, metadata=metadata))
        
        return docs
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)

def get_retriever(model_name='default'):
    """Return a LangChain compatible retriever for chat messages."""
    # Check if the necessary files exist first
    paths = get_model_paths(model_name)
    
    # Verify file existence before attempting to create retriever
    if not os.path.exists(paths['faiss_index']):
        print(f"FAISS index not found for model {model_name} at {paths['faiss_index']}")
        # Try to build the index if memory.json exists
        if os.path.exists(paths['chat_history']):
            try:
                print(f"Attempting to build index for model {model_name}")
                build_and_save_index(model_name)
                print(f"Successfully built index for model {model_name}")
            except Exception as build_err:
                print(f"Failed to build index for model {model_name}: {build_err}")
                return None
        else:
            print(f"Cannot build index: memory.json not found for model {model_name}")
            return None
            
    if not os.path.exists(paths['index_map']):
        print(f"Index map not found for model {model_name} at {paths['index_map']}")
        return None
        
    # Create the retriever
    try:
        retriever = FaissIndexRetriever(model_name)
        print(f"Successfully created retriever for model {model_name}")
        return retriever
    except Exception as e:
        print(f"Error creating retriever for model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    query = 'lordor'
    fewshot_context = get_fewshot_qa_context(query, top_k=5, format_style='qa')
    print(fewshot_context)
