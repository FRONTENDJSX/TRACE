#!/usr/bin/env python3
"""
TRACE Example Script
Demonstrates basic usage of the TRACE facial recognition system.
"""

import os
import sys
from dotenv import load_dotenv
from TRACE import TRACE
import cv2

def main():
    """Main example function."""
    print("🔍 TRACE Example - Facial Recognition System")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    gemini_key = os.getenv('GEMINI_API_KEY')
    if not gemini_key:
        print("❌ Error: GEMINI_API_KEY not found in environment")
        print("   Please set your Gemini API key in the .env file")
        print("   Copy env.example to .env and add your API key")
        return False
    
    # Initialize TRACE system
    print("🚀 Initializing TRACE system...")
    try:
        trace = TRACE(gemini_api_key=gemini_key)
        print("✅ TRACE system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TRACE: {e}")
        return False
    
    # Test camera
    print("\n📷 Testing camera...")
    if not trace.test_camera():
        print("❌ Camera test failed. Please check your camera connection.")
        return False
    
    # Show menu
    while True:
        print("\n" + "=" * 50)
        print("TRACE Example Menu:")
        print("1. Register a new person")
        print("2. Identify a person")
        print("3. View all identities")
        print("4. Test recognition (85% threshold)")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ").strip()
        
        if choice == '1':
            register_person(trace)
        elif choice == '2':
            identify_person(trace)
        elif choice == '3':
            view_identities(trace)
        elif choice == '4':
            test_recognition(trace)
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def register_person(trace):
    """Register a new person."""
    print("\n📝 Register New Person")
    print("-" * 30)
    
    name = input("Enter person's name: ").strip()
    if not name:
        print("❌ Name is required")
        return
    
    nickname = input("Enter nickname (optional): ").strip() or None
    
    print(f"\n📸 Capturing images for {name}...")
    print("Position your face in the camera and press ENTER when ready...")
    
    input("Press ENTER to start registration...")
    
    # Capture image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not capture image")
        return
    
    try:
        trace_id, confidence, metadata = trace.register_person(frame, name, nickname)
        print(f"✅ Registration successful!")
        print(f"   Trace ID: {trace_id}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Name: {name}")
        if nickname:
            print(f"   Nickname: {nickname}")
    except Exception as e:
        print(f"❌ Registration failed: {e}")

def identify_person(trace):
    """Identify a person."""
    print("\n🔍 Identify Person")
    print("-" * 30)
    
    print("Position your face in the camera and press ENTER when ready...")
    input("Press ENTER to capture image...")
    
    # Capture image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not capture image")
        return
    
    try:
        trace_id, confidence, metadata = trace.scan_person(frame)
        
        if trace_id:
            identity = trace.get_identity(trace_id)
            name = identity.get('name', 'Unknown')
            nickname = identity.get('nickname')
            
            print(f"✅ Person identified!")
            print(f"   Trace ID: {trace_id}")
            print(f"   Name: {name}")
            if nickname:
                print(f"   Nickname: {nickname}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Last seen: {metadata.get('last_seen', 'Unknown')}")
        else:
            best_similarity = metadata.get('best_similarity', 0.0)
            print(f"❌ Person not recognized")
            print(f"   Best similarity: {best_similarity:.2f}")
            print(f"   Threshold: 0.85")
    except Exception as e:
        print(f"❌ Identification failed: {e}")

def view_identities(trace):
    """View all identities."""
    print("\n👥 All Identities")
    print("-" * 30)
    
    identities = trace.list_identities()
    
    if not identities:
        print("❌ No identities found")
        return
    
    print(f"Found {len(identities)} identity(ies):")
    print()
    
    for i, identity in enumerate(identities, 1):
        print(f"{i}. {identity.get('name', 'Unknown')}")
        if identity.get('nickname'):
            print(f"   Nickname: {identity.get('nickname')}")
        print(f"   Trace ID: {identity.get('trace_id')}")
        print(f"   Observations: {identity.get('observation_count', 0)}")
        print(f"   Last seen: {identity.get('recognition_metadata', {}).get('last_seen', 'Unknown')}")
        print(f"   Confidence: {identity.get('recognition_metadata', {}).get('confidence_score', 0.0):.2f}")
        print()

def test_recognition(trace):
    """Test recognition system."""
    print("\n🧪 Recognition Test")
    print("-" * 30)
    
    print("This will test the recognition system with an 85% similarity threshold.")
    print("Position your face in the camera and press ENTER when ready...")
    
    input("Press ENTER to start test...")
    
    try:
        success = trace.person_test()
        if success:
            print("✅ Recognition test passed!")
        else:
            print("❌ Recognition test failed")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
