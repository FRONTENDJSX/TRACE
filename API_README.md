# TRACE API - Standalone Face Recognition API

A private REST API for the TRACE (Technical Recognition and Anatomical Character Engine) face recognition system.

## 🚀 Features

- **Face Identification**: Identify registered people from images
- **Person Registration**: Register new people with names and nicknames
- **Identity Management**: List, view, and delete identities
- **Recognition Testing**: Test the recognition system
- **Statistics**: Get system statistics and metrics

## 📋 Requirements

- Python 3.8+
- Google Gemini API key
- OpenCV compatible camera (for testing)

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r trace_api_requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file with your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Run the API server**:
   ```bash
   python trace_api.py
   ```

The API will be available at `http://localhost:5000`

## 📚 API Endpoints

### Health Check
- **GET** `/api/health` - Check API health status

### Identity Management
- **GET** `/api/identities` - List all registered identities
- **GET** `/api/identity/<trace_id>` - Get specific identity details
- **DELETE** `/api/identity/<trace_id>` - Delete an identity

### Face Recognition
- **POST** `/api/identify` - Identify a person from an image
- **POST** `/api/register` - Register a new person
- **POST** `/api/test` - Test the recognition system

### System Information
- **GET** `/api/stats` - Get system statistics

## 🔧 API Usage Examples

### Identify a Person
```bash
curl -X POST http://localhost:5000/api/identify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  }'
```

### Register a New Person
```bash
curl -X POST http://localhost:5000/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "name": "John Doe",
    "nickname": "Johnny"
  }'
```

### List All Identities
```bash
curl http://localhost:5000/api/identities
```

## 🌐 Web Client

A simple web client is included (`trace_web_client.html`) for testing the API:

1. Open `trace_web_client.html` in your browser
2. Upload an image
3. Test identification and registration features

## 📊 Response Formats

### Successful Identification
```json
{
  "success": true,
  "identified": true,
  "trace_id": "user_0001",
  "name": "John Doe",
  "nickname": "Johnny",
  "confidence": 0.95,
  "last_seen": "2025-01-23T20:30:00Z",
  "observation_count": 5
}
```

### Person Not Recognized
```json
{
  "success": true,
  "identified": false,
  "message": "Person not recognized",
  "best_similarity": 0.45,
  "threshold": 0.85
}
```

### Error Response
```json
{
  "error": "Error message description"
}
```

## 🔒 Security Notes

- This is a **private API** - not intended for public use
- No authentication is implemented (add as needed)
- API runs on all interfaces (0.0.0.0) - restrict as needed
- Image data is processed in memory and not stored permanently

## 🐛 Troubleshooting

### Common Issues

1. **"TRACE system not initialized"**
   - Check that `GEMINI_API_KEY` is set in your `.env` file
   - Ensure the API key is valid

2. **"Failed to decode image"**
   - Ensure image is in base64 format
   - Check that image data is valid

3. **"No face detected"**
   - Ensure the image contains a clear, well-lit face
   - Try different angles or lighting

### Debug Mode

The API runs in debug mode by default. To disable:
```python
app.run(host='0.0.0.0', port=5000, debug=False)
```

## 📈 Performance

- **Recognition Speed**: ~2-5 seconds per image
- **Accuracy**: 85% similarity threshold
- **Concurrent Requests**: Limited by Gemini API rate limits
- **Memory Usage**: ~100-200MB base + image processing

## 🔄 Integration

The API can be integrated with:
- Web applications
- Mobile apps
- Desktop applications
- IoT devices
- Security systems

## 📝 License

Private use only. Not for public distribution.
