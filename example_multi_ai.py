"""
TRACE Multi-AI Example
Demonstrates how to use TRACE with different AI providers
"""

import os
import cv2
import numpy as np
from TRACE_multi_ai import TRACE
from config import get_config

def main():
    """Main example function"""
    print("=== TRACE Multi-AI Example ===")
    print("This example demonstrates how to use TRACE with different AI providers")
    print()
    
    # Load configuration
    config = get_config()
    
    # Show available providers
    print("🔧 Available AI Providers:")
    available_providers = config.get_available_providers()
    for provider in available_providers:
        provider_config = config.get_ai_provider_config(provider)
        print(f"   ✅ {provider}: {provider_config.get('enabled', False)}")
    
    print(f"\n🎯 Primary Provider: {config.get_primary_provider()}")
    print()
    
    # Initialize TRACE system
    print("🚀 Initializing TRACE system...")
    trace = TRACE()
    
    print(f"✅ TRACE initialized with {type(trace.ai_provider).__name__}")
    print(f"   Provider Available: {trace.ai_provider.is_available()}")
    print()
    
    # Test AI provider
    print("🧪 Testing AI provider...")
    if trace.test_ai_provider():
        print("✅ AI provider test passed")
    else:
        print("❌ AI provider test failed")
        print("   Please check your configuration")
        return
    
    print()
    
    # Example 1: Register a person with different providers
    print("=== Example 1: Register a Person ===")
    
    # Create a sample image (in real usage, you'd capture from camera)
    sample_image = create_sample_image()
    
    # Register a person
    try:
        trace_id, confidence, metadata = trace.register_person(
            sample_image, 
            name="John Doe", 
            nickname="Johnny"
        )
        print(f"✅ Registered person: {trace_id}")
        print(f"   Name: John Doe (Johnny)")
        print(f"   Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return
    
    print()
    
    # Example 2: Switch AI providers
    print("=== Example 2: Switch AI Providers ===")
    
    available_providers = trace.get_available_providers()
    print(f"Available providers: {', '.join(available_providers)}")
    
    # Try to switch to a different provider if available
    for provider in available_providers:
        if provider != type(trace.ai_provider).__name__.lower().replace('provider', ''):
            print(f"🔄 Switching to {provider}...")
            try:
                trace.switch_ai_provider(provider)
                print(f"✅ Switched to {provider}")
                break
            except Exception as e:
                print(f"❌ Failed to switch to {provider}: {e}")
                continue
    
    print()
    
    # Example 3: Test recognition
    print("=== Example 3: Test Recognition ===")
    
    # Create another sample image
    test_image = create_sample_image()
    
    try:
        trace_id, confidence, metadata = trace.scan_person(test_image)
        
        if trace_id:
            print(f"✅ Person recognized: {trace_id}")
            print(f"   Confidence: {confidence:.2f}")
            
            # Get identity details
            identity = trace.get_identity(trace_id)
            if identity:
                print(f"   Name: {identity.get('name')}")
                print(f"   Nickname: {identity.get('nickname')}")
                print(f"   Observations: {identity.get('observation_count', 0)}")
        else:
            print("❌ Person not recognized")
            print(f"   Best similarity: {metadata.get('best_similarity', 0.0):.2f}")
    except Exception as e:
        print(f"❌ Recognition failed: {e}")
    
    print()
    
    # Example 4: List identities
    print("=== Example 4: List Identities ===")
    
    identities = trace.list_identities()
    print(f"📊 Found {len(identities)} identities:")
    
    for identity in identities:
        print(f"   👤 {identity.get('trace_id')}: {identity.get('name')}")
        print(f"      Nickname: {identity.get('nickname', 'None')}")
        print(f"      Observations: {identity.get('observation_count', 0)}")
        print(f"      Last seen: {identity.get('recognition_metadata', {}).get('last_seen', 'Unknown')}")
        print()
    
    # Example 5: Get statistics
    print("=== Example 5: System Statistics ===")
    
    stats = trace.get_statistics()
    print(f"📊 System Statistics:")
    print(f"   Total identities: {stats['total_identities']}")
    print(f"   Total observations: {stats['total_observations']}")
    print(f"   AI Provider: {stats['ai_provider']}")
    print(f"   Provider available: {stats['ai_provider_available']}")
    print(f"   Data file: {stats['data_file']}")
    
    print()
    print("🎉 Multi-AI example completed!")

def create_sample_image():
    """Create a sample image for testing"""
    # Create a simple test image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some simple features to make it look like a face
    # This is just for demonstration - in real usage, you'd capture from camera
    cv2.circle(image, (320, 200), 50, (255, 255, 255), -1)  # Face
    cv2.circle(image, (300, 180), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(image, (340, 180), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(image, (320, 220), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    return image

def demonstrate_provider_switching():
    """Demonstrate switching between different AI providers"""
    print("=== Provider Switching Demo ===")
    
    # Initialize TRACE
    trace = TRACE()
    
    # Get available providers
    available_providers = trace.get_available_providers()
    print(f"Available providers: {', '.join(available_providers)}")
    
    # Test each provider
    for provider in available_providers:
        print(f"\n🔄 Testing {provider} provider...")
        
        try:
            # Switch to provider
            trace.switch_ai_provider(provider)
            
            # Test the provider
            if trace.test_ai_provider():
                print(f"✅ {provider} provider is working")
            else:
                print(f"❌ {provider} provider test failed")
        except Exception as e:
            print(f"❌ {provider} provider error: {e}")
    
    print("\n🎯 Provider switching demo completed!")

if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run provider switching demo
    # demonstrate_provider_switching()
