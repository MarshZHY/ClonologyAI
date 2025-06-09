# Clonology.AI

<div align="center">

![ClonologyAI Logo](https://img.shields.io/badge/ClonologyAI-Advanced%20Memory%20Cloning-blue?style=for-the-badge)

**Transform Your Chat History into Intelligent AI Clones**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.0.267-orange.svg)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[🚀 Quick Start](#quick-start) • [📖 Documentation](#documentation) • [🎯 Features](#features) • [🛠️ Installation](#installation) • [🤝 Contributing](#contributing)

</div>

---

## 🌟 Overview

ClonologyAI is an advanced AI platform that creates personalized AI models from your chat history. Using state-of-the-art memory cloning technology, it learns your unique communication style, personality traits, and conversational patterns to create an AI clone that speaks just like you.

### ✨ Key Highlights

- 🧠 **Memory-Aware Conversations** - Your AI clone remembers past interactions and context
- 🎭 **Personality Cloning** - Captures your unique communication style and mannerisms  
- 📊 **Human Evaluation** - Compare AI clone performance against baseline models
- 🔧 **Customizable Processing** - Adjustable memory precision and processing speed

---

## 🎯 Features

### 🏗️ **AI Clone Creation**
- Upload your chat history (JSON format)
- Select participant to clone
- Configurable memory processing (Precise/Balanced/Fast modes)
- Automatic participant detection and analysis

### 💬 **Direct Chat Interface**
- Natural conversation with your AI clone
- Real-time response generation
- Memory-aware context understanding
- Personalized communication style matching

### 📈 **Performance Evaluation**
- Human evaluation system with A/B testing
- Compare AI clone vs baseline models
- Multiple evaluation types:
  - 💬 Natural Chat
  - 🧠 Memory Recall  
  - 👤 Personalization
  - ❤️ Emotional Support
- Real-time analytics and performance metrics

### 🔍 **Advanced Analytics**
- Comprehensive chat analysis and statistics
- Message distribution and pattern recognition
- Emoji usage and communication style analysis
- Question pattern identification

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (or compatible LLM API)
- 4GB+ RAM recommended for processing

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ClonologyAI.git
cd ClonologyAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
TYPHOON_API_KEY=your_typhoon_api_key_here  # Optional: for Thai language support
```

Or update the [`config.json`](config.json) file:

```json
{
  "api_key": "your_api_key_here"
}
```

### 4. Initialize Model Structure

```bash
python app.py
```

The application will automatically create the necessary model directories and files.

---

## 🚀 Quick Start

### 1. Start the Application

```bash
python app.py
```

Visit `http://localhost:5000` to access the web interface.

### 2. Create Your AI Clone

1. **Upload Data**: Go to [`/setup`](http://localhost:5000/setup) and upload your chat history JSON file
2. **Select Participant**: Choose which participant from your chat to clone
3. **Configure Processing**: Adjust memory processing settings (1-10 scale)
4. **Process**: Wait for the AI clone creation to complete

### 3. Start Chatting

1. **Direct Chat**: Visit [`/direct_chat`](http://localhost:5000/direct_chat) to chat with your AI clone
2. **Evaluation**: Use [`/index`](http://localhost:5000/index) to compare your AI clone against baseline models

---

## 📊 Project Structure

```
ClonologyAI/
├── 📁 static/                 # Frontend assets
│   ├── styles.css            # Modern UI styling
│   ├── script.js             # Main JavaScript functionality
│   └── script.js.new         # Updated JavaScript features
├── 📁 templates/             # HTML templates
│   ├── home.html             # Landing page
│   ├── setup.html            # AI clone creation
│   ├── direct_chat.html      # Chat interface
│   └── index.html            # Evaluation interface
├── 📁 model/                 # AI model storage
│   └── default/              # Default model directory
├── app.py                    # Main Flask application
├── modeling.py               # AI model processing
├── chat_rag.py              # RAG implementation
├── chat_eda.py              # Chat analysis
├── requirements.txt          # Python dependencies
├── config.json              # Configuration
└── README.md                # This file
```

---

## 🔧 Configuration

### Memory Processing Settings

The [`modeling.py`](modeling.py) module supports various configuration options:

- **Chunk Size**: Adjustable message chunking (1000-5000 tokens)
- **Processing Modes**: 
  - Precise (1-3): Detailed analysis, slower processing
  - Balanced (4-7): Optimal speed/accuracy balance  
  - Fast (8-10): Quick processing, larger chunks

### LLM Configuration

Update the [`initialize_llm()`](modeling.py) function to use different language models:

```python
def initialize_llm():
    return ChatOpenAI(
        model_name="gpt-4",  # or your preferred model
        temperature=0.7,
        max_tokens=2000
    )
```

---

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | Get available AI models |
| `/api/chat` | POST | Chat with AI clone |
| `/api/evaluate` | POST | Evaluate AI responses |
| `/api/process_model` | POST | Start model processing |
| `/api/process_status` | GET | Get processing status |

---

## 🧪 Testing

### Chat Analysis

Run comprehensive chat analysis:

```bash
python chat_eda.py
```

This generates detailed analytics in the `analysis_results/` directory.

### Model Evaluation

The evaluation system provides multiple testing scenarios:

1. **Natural Chat**: Test conversational abilities
2. **Memory Recall**: Evaluate memory and context understanding  
3. **Personalization**: Assess personality matching
4. **Emotional Support**: Test empathy and emotional responses

---

## 📚 Dependencies

Key dependencies from [`requirements.txt`](requirements.txt):

```txt
flask==2.3.3
langchain==0.0.267
openai==0.27.8
python-dotenv==1.0.0
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0
```

For Thai language support:
```txt
pythainlp>=3.0.0
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** for the powerful LLM framework
- **Hugging Face** for embedding models
- **FAISS** for efficient similarity search
- **Flask** for the web framework

---

## 📞 Support

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/ClonologyAI/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/yourusername/ClonologyAI/discussions)
- 📧 **Contact**: your.email@example.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the ClonologyAI Team

</div>
