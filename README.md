# TRACE - Technical Recognition and Anatomical Character Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/frontendjsx/TRACE)

**TRACE** is the first **open-source adaptive AI identity recognition framework**
 designed for *thinking AI systems*, not just authentication systems.


## 🌟 Features

- **Multi-AI Support**: Works with Gemini, OpenAI, Claude, and Local LLMs (Ollama)
- **Anatomical Intelligence**: Uses 60+ facial measurements for reliable recognition
- **AI-Powered Analysis**: Leverages multiple AI providers for advanced face analysis
- **Adaptive Learning**: Gradually adapts to appearance changes over time
- **Multi-Pose Support**: Works with front, profile, and angled views
- **REST API**: Complete REST API for integration with other systems
- **Web Interface**: Simple web client for testing and demonstration
- **Privacy-Focused**: No cloud storage of biometric data
- **Local LLM Support**: Run completely offline with Ollama

## 🤖 Supported AI Providers

### Cloud APIs
- **Google Gemini**: `gemini-2.5-flash` (default)
- **OpenAI GPT-4 Vision**: `gpt-4-vision-preview`
- **Anthropic Claude**: `claude-3-5-sonnet-20241022`

### Local LLMs
- **Ollama**: Any vision-capable model (e.g., `llava:7b`, `llava:13b`)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- At least one AI provider API key or local LLM setup
- Webcam or camera for testing

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/TRACE.git
   cd TRACE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the demo**:
   ```bash
   python TRACE_multi_ai.py
   ```

### API Server

To run the REST API server:

```bash
python trace_api_multi_ai.py
```

The API will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | Primary AI provider | `gemini` |
| `GEMINI_API_KEY` | Google Gemini API key | None |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | None |
| `OLLAMA_ENABLED` | Enable Ollama local LLM | `false` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llava:7b` |
| `TRACE_DATA_FILE` | Path to identity database | `TRACE.json` |
| `TRACE_IMAGES_DIR` | Directory for reference images | `images/` |

### AI Provider Configuration

#### Google Gemini
```bash
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

#### OpenAI GPT-4 Vision
```bash
AI_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-vision-preview
```

#### Anthropic Claude
```bash
AI_PROVIDER=claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
```

#### Local Ollama
```bash
AI_PROVIDER=ollama
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llava:7b
```

## 📚 Usage

### Basic Python Integration

```python
from TRACE_multi_ai import TRACE

# Initialize with specific AI provider
trace = TRACE(ai_provider="gemini")

# Or let it auto-detect from configuration
trace = TRACE()

# Register a new person
trace_id, confidence, metadata = trace.register_person(
    image, 
    name="John Doe", 
    nickname="Johnny"
)

# Identify a person
trace_id, confidence, metadata = trace.scan_person(image)
if trace_id:
    print(f"Recognized: {trace_id} with confidence {confidence:.2f}")

# Switch AI providers
trace.switch_ai_provider("openai")
trace.switch_ai_provider("claude")
trace.switch_ai_provider("ollama")

# Test AI provider
if trace.test_ai_provider():
    print("AI provider is working!")
```

### REST API Usage

#### List Available Providers
```bash
curl -X GET http://localhost:5000/api/providers
```

#### Switch AI Provider
```bash
curl -X POST http://localhost:5000/api/providers/openai
```

#### Test AI Provider
```bash
curl -X POST http://localhost:5000/api/test-provider
```

#### Register a Person
```bash
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "name": "John Doe",
    "nickname": "Johnny"
  }'
```

#### Identify a Person
```bash
curl -X POST http://localhost:5000/api/identify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  }'
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with AI provider info |
| GET | `/api/providers` | List available AI providers |
| POST | `/api/providers/<name>` | Switch AI provider |
| POST | `/api/test-provider` | Test current AI provider |
| GET | `/api/identities` | List all identities |
| GET | `/api/identity/<id>` | Get identity details |
| POST | `/api/register` | Register new person |
| POST | `/api/identify` | Identify person from image |
| POST | `/api/test` | Test recognition system |
| DELETE | `/api/identity/<id>` | Delete identity |
| GET | `/api/stats` | Get system statistics |

## 🔒 Privacy & Security

- **Local Processing**: All face analysis happens locally (with Ollama)
- **No Cloud Storage**: Biometric data never leaves your system
- **Encrypted Storage**: Identity data can be encrypted at rest
- **API Key Security**: Store API keys in environment variables only
- **Offline Mode**: Use Ollama for completely offline operation

## 🛠️ Development

### Running Tests

```bash
python example_multi_ai.py
```

### Building Documentation

```bash
pip install sphinx
cd docs/
make html
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📈 Performance

- **Recognition Speed**: 2-5 seconds per image (varies by provider)
- **Accuracy**: 85%+ similarity threshold
- **Memory Usage**: ~100-200MB base + image processing
- **Concurrent Requests**: Limited by API rate limits (cloud providers)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Gemini API for advanced AI capabilities
- OpenAI for GPT-4 Vision API
- Anthropic for Claude API
- Ollama for local LLM support
- OpenCV community for computer vision tools
- Flask community for the web framework
- All contributors who help improve TRACE

## 📞 Support

- 💬 Discord: [Join our community](https://discord.gg/trace-ai)
- 🐛 Issues: [GitHub Issues](https://github.com/frontendjsx/TRACE/issues)

---

**Made with ❤️ by the TRACE team**
