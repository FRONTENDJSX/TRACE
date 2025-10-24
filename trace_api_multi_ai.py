"""
TRACE API - Multi-AI Provider Face Recognition API
Supports Gemini, OpenAI, Claude, and Local LLMs
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
from TRACE_multi_ai import TRACE
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for web clients

# Initialize TRACE system
trace_system = None

def init_trace():
    """Initialize TRACE system with AI provider"""
    global trace_system
    try:
        from config import get_config
        config = get_config()
        
        # Get AI provider from environment or config
        ai_provider = os.getenv('AI_PROVIDER', config.get_primary_provider())
        
        # Initialize TRACE with the specified provider
        trace_system = TRACE(ai_provider=ai_provider)
        
        print(f"✅ TRACE API initialized with {type(trace_system.ai_provider).__name__}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize TRACE: {e}")
        return False

def decode_image(image_data):
    """Decode base64 image data to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "TRACE Multi-AI API",
        "version": "2.0.0",
        "ai_provider": type(trace_system.ai_provider).__name__ if trace_system else "Unknown",
        "ai_provider_available": trace_system.ai_provider.is_available() if trace_system else False
    })

@app.route('/api/providers', methods=['GET'])
def list_providers():
    """List available AI providers"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        available_providers = trace_system.get_available_providers()
        current_provider = type(trace_system.ai_provider).__name__
        
        return jsonify({
            "success": True,
            "available_providers": available_providers,
            "current_provider": current_provider,
            "current_provider_available": trace_system.ai_provider.is_available()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/providers/<provider_name>', methods=['POST'])
def switch_provider(provider_name):
    """Switch to a different AI provider"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        available_providers = trace_system.get_available_providers()
        
        if provider_name not in available_providers:
            return jsonify({
                "error": f"Provider '{provider_name}' not available",
                "available_providers": available_providers
            }), 400
        
        # Switch provider
        trace_system.switch_ai_provider(provider_name)
        
        return jsonify({
            "success": True,
            "message": f"Switched to {provider_name} provider",
            "current_provider": type(trace_system.ai_provider).__name__,
            "provider_available": trace_system.ai_provider.is_available()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-provider', methods=['POST'])
def test_provider():
    """Test the current AI provider"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        success = trace_system.test_ai_provider()
        
        return jsonify({
            "success": True,
            "test_passed": success,
            "provider": type(trace_system.ai_provider).__name__,
            "provider_available": trace_system.ai_provider.is_available(),
            "message": "AI provider test completed"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/identities', methods=['GET'])
def list_identities():
    """List all registered identities"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        identities = trace_system.list_identities()
        
        # Format response
        formatted_identities = []
        for identity in identities:
            formatted_identities.append({
                "trace_id": identity.get('trace_id'),
                "name": identity.get('name'),
                "nickname": identity.get('nickname'),
                "observation_count": identity.get('observation_count', 0),
                "last_seen": identity.get('recognition_metadata', {}).get('last_seen'),
                "confidence": identity.get('recognition_metadata', {}).get('confidence_score', 0.0)
            })
        
        return jsonify({
            "success": True,
            "identities": formatted_identities,
            "count": len(formatted_identities)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/identify', methods=['POST'])
def identify_person():
    """Identify a person from an image"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        image = decode_image(data['image'])
        
        # Identify person
        trace_id, confidence, metadata = trace_system.scan_person(image)
        
        if trace_id is None:
            # No match found
            best_similarity = metadata.get('best_similarity', 0.0)
            return jsonify({
                "success": True,
                "identified": False,
                "message": "Person not recognized",
                "best_similarity": best_similarity,
                "threshold": 0.85
            })
        
        # Get identity details
        identity = trace_system.get_identity(trace_id)
        if not identity:
            return jsonify({
                "success": True,
                "identified": False,
                "message": "Identity data not found"
            })
        
        return jsonify({
            "success": True,
            "identified": True,
            "trace_id": trace_id,
            "name": identity.get('name'),
            "nickname": identity.get('nickname'),
            "confidence": confidence,
            "last_seen": identity.get('recognition_metadata', {}).get('last_seen'),
            "observation_count": identity.get('observation_count', 0)
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_person():
    """Register a new person"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        if 'name' not in data:
            return jsonify({"error": "Name is required"}), 400
        
        # Decode image
        image = decode_image(data['image'])
        
        # Register person
        trace_id, confidence, metadata = trace_system.register_person(
            image, 
            data['name'], 
            data.get('nickname')
        )
        
        # Get identity details
        identity = trace_system.get_identity(trace_id)
        
        return jsonify({
            "success": True,
            "trace_id": trace_id,
            "name": identity.get('name'),
            "nickname": identity.get('nickname'),
            "confidence": confidence,
            "message": "Person registered successfully"
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/identity/<trace_id>', methods=['GET'])
def get_identity(trace_id):
    """Get specific identity details"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        identity = trace_system.get_identity(trace_id)
        if not identity:
            return jsonify({"error": "Identity not found"}), 404
        
        return jsonify({
            "success": True,
            "identity": {
                "trace_id": identity.get('trace_id'),
                "name": identity.get('name'),
                "nickname": identity.get('nickname'),
                "observation_count": identity.get('observation_count', 0),
                "last_seen": identity.get('recognition_metadata', {}).get('last_seen'),
                "confidence": identity.get('recognition_metadata', {}).get('confidence_score', 0.0),
                "age_estimation": identity.get('age_estimation', {}),
                "image_count": len(identity.get('image_references', []))
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/identity/<trace_id>', methods=['DELETE'])
def delete_identity(trace_id):
    """Delete an identity"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        success = trace_system.delete_identity(trace_id)
        if not success:
            return jsonify({"error": "Identity not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Identity deleted successfully"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test_recognition():
    """Test the recognition system"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode image
        image = decode_image(data['image'])
        
        # Test recognition using the person_test function
        from TRACE_multi_ai import person_test
        success = person_test(trace_system)
        
        return jsonify({
            "success": True,
            "test_passed": success,
            "message": "Recognition test completed"
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    try:
        if not trace_system:
            return jsonify({"error": "TRACE system not initialized"}), 500
        
        stats = trace_system.get_statistics()
        
        return jsonify({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting TRACE Multi-AI API Server...")
    
    # Initialize TRACE
    if not init_trace():
        print("❌ Failed to initialize TRACE system. Exiting.")
        exit(1)
    
    print("🌐 TRACE Multi-AI API Server running on http://localhost:5000")
    print("📚 API Documentation:")
    print("   GET  /api/health - Health check")
    print("   GET  /api/providers - List available AI providers")
    print("   POST /api/providers/<name> - Switch AI provider")
    print("   POST /api/test-provider - Test current AI provider")
    print("   GET  /api/identities - List all identities")
    print("   POST /api/identify - Identify person from image")
    print("   POST /api/register - Register new person")
    print("   GET  /api/identity/<id> - Get identity details")
    print("   DELETE /api/identity/<id> - Delete identity")
    print("   POST /api/test - Test recognition system")
    print("   GET  /api/stats - Get system statistics")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
