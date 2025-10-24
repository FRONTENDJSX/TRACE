"""
TRACE - Technical Recognition and Anatomical Character Engine
Multi-AI Provider Version - Supports Gemini, OpenAI, Claude, and Local LLMs

CONSOLIDATED VERSION - All TRACE functionality with multiple AI providers
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import numpy as np
import cv2
import base64
import io

# Import AI providers and configuration
from ai_providers import AIProvider, AIProviderFactory
from config import get_config

class TRACE:
    def __init__(self, data_file: str = None, ai_provider: str = None, **provider_kwargs):
        """
        Initialize TRACE system with multiple AI providers
        
        Args:
            data_file: Path to TRACE.json file
            ai_provider: AI provider to use ('gemini', 'openai', 'claude', 'ollama')
            **provider_kwargs: Provider-specific configuration
        """
        # Load configuration
        self.config = get_config()
        
        # Set data file
        self.data_file = data_file or self.config.get("data_file", "TRACE.json")
        self.next_id = 1
        
        # Create images directory if it doesn't exist
        images_dir = self.config.get("images_dir", "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Initialize AI provider
        self.ai_provider = self._initialize_ai_provider(ai_provider, **provider_kwargs)
        
        # Load existing identities
        self.identities = self._load_identities()
        self._update_next_id()
        
        # Recognition thresholds from config
        self.ANATOMICAL_THRESHOLD = self.config.get("anatomical_threshold", 0.85)
        self.ANATOMICAL_WEIGHT = self.config.get("anatomical_weight", 0.95)
        self.ACCESSORY_WEIGHT = self.config.get("accessory_weight", 0.02)
        self.EXPRESSION_WEIGHT = self.config.get("expression_weight", 0.0)
    
    def _initialize_ai_provider(self, ai_provider: str = None, **provider_kwargs) -> AIProvider:
        """Initialize AI provider"""
        # Determine provider to use
        if ai_provider:
            provider_type = ai_provider
        else:
            provider_type = self.config.get_primary_provider()
        
        # Get provider configuration
        provider_config = self.config.get_ai_provider_config(provider_type)
        
        # Merge with provided kwargs
        config = {**provider_config, **provider_kwargs}
        
        # Create provider
        try:
            provider = AIProviderFactory.create_provider(provider_type, **config)
            if provider.is_available():
                print(f"✅ Initialized {provider_type} AI provider")
                return provider
            else:
                print(f"❌ {provider_type} provider is not available")
                return self._get_fallback_provider()
        except Exception as e:
            print(f"❌ Failed to initialize {provider_type} provider: {e}")
            return self._get_fallback_provider()
    
    def _get_fallback_provider(self) -> AIProvider:
        """Get fallback provider from available providers"""
        available_providers = self.config.get_available_providers()
        
        for provider_type in available_providers:
            try:
                provider_config = self.config.get_ai_provider_config(provider_type)
                provider = AIProviderFactory.create_provider(provider_type, **provider_config)
                if provider.is_available():
                    print(f"✅ Using fallback provider: {provider_type}")
                    return provider
            except Exception as e:
                print(f"❌ Fallback provider {provider_type} failed: {e}")
                continue
        
        # If no provider is available, create a dummy provider
        print("❌ No AI providers are available. Face analysis will be limited.")
        return DummyProvider()
    
    def _load_identities(self) -> Dict[str, Dict]:
        """Load existing identities from TRACE.json file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_identities(self):
        """Save identities to TRACE.json file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.identities, f, indent=2)
    
    def _update_next_id(self):
        """Update the next available ID number"""
        if self.identities:
            max_id = max(int(identity_id.split('_')[1]) for identity_id in self.identities.keys())
            self.next_id = max_id + 1
    
    def _generate_trace_id(self) -> str:
        """Generate next unique trace ID"""
        trace_id = f"user_{self.next_id:04d}"
        self.next_id += 1
        return trace_id
    
    def _save_image(self, image: np.ndarray, trace_id: str, scan_number: int) -> str:
        """Save image and return path"""
        images_dir = self.config.get("images_dir", "images")
        image_dir = os.path.join(images_dir, trace_id)
        os.makedirs(image_dir, exist_ok=True)
        
        filename = f"scan_{scan_number:03d}.jpg"
        image_path = os.path.join(image_dir, filename)
        
        cv2.imwrite(image_path, image)
        return image_path
    
    def _analyze_face_with_ai(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face using configured AI provider"""
        print(f"🤖 Analyzing face with {type(self.ai_provider).__name__}")
        try:
            return self.ai_provider.analyze_face(image)
        except Exception as e:
            print(f"AI analysis error: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when AI provider is not available"""
        print("❌ AI provider is required for face analysis. Please configure an AI provider.")
        return {}
    
    def _calculate_anatomical_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate anatomical similarity between two feature sets using 60+ features"""
        numeric1 = features1.get("face_numeric", {})
        numeric2 = features2.get("face_numeric", {})
        
        print(f"🔍 Comparing features:")
        print(f"   Features1 keys: {list(numeric1.keys())[:5]}... (total: {len(numeric1)})")
        print(f"   Features2 keys: {list(numeric2.keys())[:5]}... (total: {len(numeric2)})")
        
        if not numeric1 or not numeric2:
            print(f"   ❌ Missing features: numeric1={bool(numeric1)}, numeric2={bool(numeric2)}")
            return 0.0
        
        # Calculate similarity for each measurement with weighted importance
        similarities = []
        weights = {
            # Core facial structure (highest weight)
            'face_height_in': 1.0, 'face_width_in': 1.0, 'face_length_in': 1.0,
            'jaw_length_in': 1.0, 'jaw_width_in': 1.0, 'jaw_angle_deg': 1.0,
            'cheekbone_width_in': 1.0, 'cheekbone_height_in': 1.0,
            'interocular_distance_in': 1.0,
            
            # Eye features (high weight)
            'eye_width_left_in': 0.9, 'eye_width_right_in': 0.9,
            'eye_height_left_in': 0.9, 'eye_height_right_in': 0.9,
            'eye_angle_left_deg': 0.9, 'eye_angle_right_deg': 0.9,
            
            # Nose features (high weight)
            'nose_length_in': 0.9, 'nose_width_in': 0.9, 'nose_height_in': 0.9,
            'nose_angle_deg': 0.9, 'nose_bridge_width_in': 0.8,
            
            # Lip features (medium weight)
            'lip_width_in': 0.8, 'lip_height_in': 0.8, 'lip_angle_deg': 0.8,
            'lip_thickness_upper_in': 0.7, 'lip_thickness_lower_in': 0.7,
            
            # Forehead and temple (medium weight)
            'forehead_width_in': 0.7, 'forehead_height_in': 0.7,
            'temple_width_in': 0.7, 'temple_depth_in': 0.7,
            
            # Chin features (medium weight)
            'chin_width_in': 0.7, 'chin_height_in': 0.7, 'chin_angle_deg': 0.7,
            
            # Ear features (lower weight)
            'ear_width_left_in': 0.6, 'ear_width_right_in': 0.6,
            'ear_height_left_in': 0.6, 'ear_height_right_in': 0.6,
            
            # Eyebrow features (lower weight)
            'eyebrow_width_left_in': 0.6, 'eyebrow_width_right_in': 0.6,
            'eyebrow_height_left_in': 0.6, 'eyebrow_height_right_in': 0.6,
            'eyebrow_angle_left_deg': 0.6, 'eyebrow_angle_right_deg': 0.6,
            
            # Facial ratios (medium weight)
            'facial_symmetry_score': 0.8, 'face_oval_ratio': 0.7,
            'face_triangle_ratio': 0.7, 'face_square_ratio': 0.7,
            'face_heart_ratio': 0.7, 'face_diamond_ratio': 0.7, 'face_round_ratio': 0.7
        }
        
        compared_features = 0
        skipped_none = 0
        skipped_invalid = 0
        
        for key in numeric1:
            if key in numeric2:
                val1, val2 = numeric1[key], numeric2[key]
                # Skip if either value is None or not a number
                if val1 is None or val2 is None:
                    skipped_none += 1
                    if compared_features < 3:  # Show first few examples
                        print(f"   ⚠️ Skipping {key}: val1={val1}, val2={val2}")
                    continue
                try:
                    # Calculate weighted similarity
                    diff = abs(val1 - val2)
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1.0 - (diff / max_val)
                        weight = weights.get(key, 0.5)  # Default weight for unknown features
                        similarities.append(max(0, similarity) * weight)
                        compared_features += 1
                        if compared_features <= 3:  # Show first few examples
                            print(f"   ✅ Comparing {key}: {val1} vs {val2} = {similarity:.3f}")
                except (TypeError, ValueError) as e:
                    # Skip if values can't be compared
                    skipped_invalid += 1
                    if compared_features < 3:  # Show first few examples
                        print(f"   ❌ Error comparing {key}: {e}")
                    continue
        
        print(f"   📊 Stats: Compared {compared_features}, Skipped None: {skipped_none}, Skipped Invalid: {skipped_invalid}")
        
        print(f"   ✅ Compared {compared_features} features, got {len(similarities)} similarities")
        
        if not similarities:
            return 0.0
        
        # Return weighted average
        total_weight = sum(weights.get(key, 0.5) for key in numeric1 if key in numeric2)
        return sum(similarities) / total_weight if total_weight > 0 else 0.0
    
    def _update_moving_average(self, current: Dict, new: Dict, alpha: float = 0.1) -> Dict:
        """Update moving average for transient features"""
        if not current:
            return new.copy()
        
        updated = current.copy()
        for key, value in new.items():
            if isinstance(value, (int, float)):
                if key in updated:
                    updated[key] = alpha * value + (1 - alpha) * updated[key]
                else:
                    updated[key] = value
            elif isinstance(value, dict) and key in updated:
                updated[key] = self._update_moving_average(updated[key], value, alpha)
            else:
                updated[key] = value
        
        return updated
    
    def register_person(self, image: np.ndarray, name: str, nickname: Optional[str] = None) -> Tuple[str, float, Dict]:
        """
        Register a new person (explicit registration)
        
        Args:
            image: OpenCV image array
            name: Name for the person
            nickname: Optional nickname for the person
            
        Returns:
            Tuple of (trace_id, confidence, metadata)
        """
        # Analyze face
        analysis = self._analyze_face_with_ai(image)
        
        # Extract features
        stable_features = {
            "face_numeric": analysis.get("face_numeric", {}),
            "shape_text": analysis.get("shape_text", []),
            "colors": analysis.get("colors", {})
        }
        
        transient_features = {
            "expression": {
                "current": analysis.get("expression", "neutral"),
                "weight": self.EXPRESSION_WEIGHT,
                "update_method": "moving_average"
            },
            "accessories": {}
        }
        
        # Process accessories
        accessories = analysis.get("accessories", {})
        for accessory_type, data in accessories.items():
            if data.get("present", False):
                transient_features["accessories"][accessory_type] = {
                    "observed_frequency": 1.0,
                    "description": data.get("description", ""),
                    "weight": self.ACCESSORY_WEIGHT
                }
        
        # Create new identity
        return self._create_new_identity(image, stable_features, transient_features, name, nickname, analysis)

    def scan_person(self, image: np.ndarray, name: Optional[str] = None, 
                   nickname: Optional[str] = None, manual_rescan: bool = False) -> Tuple[str, float, Dict]:
        """
        Scan a person and either create new identity or match existing
        
        Args:
            image: OpenCV image array
            name: Optional name for the person
            nickname: Optional nickname for the person
            manual_rescan: If True, force update of existing identity
            
        Returns:
            Tuple of (trace_id, confidence, metadata)
        """
        # Analyze face
        analysis = self._analyze_face_with_ai(image)
        
        # Extract features
        stable_features = {
            "face_numeric": analysis.get("face_numeric", {}),
            "shape_text": analysis.get("shape_text", []),
            "colors": analysis.get("colors", {})
        }
        
        transient_features = {
            "expression": {
                "current": analysis.get("expression", "neutral"),
                "weight": self.EXPRESSION_WEIGHT,
                "update_method": "moving_average"
            },
            "accessories": {}
        }
        
        # Process accessories
        accessories = analysis.get("accessories", {})
        for accessory_type, data in accessories.items():
            if data.get("present", False):
                transient_features["accessories"][accessory_type] = {
                    "observed_frequency": 1.0,
                    "description": data.get("description", ""),
                    "weight": self.ACCESSORY_WEIGHT
                }
        
        # Find best match
        best_match_id = None
        best_similarity = 0.0
        
        for trace_id, identity in self.identities.items():
            if manual_rescan and identity.get("name") == name:
                # Manual rescan - update existing identity
                return self._update_identity(trace_id, image, stable_features, transient_features)
            
            similarity = self._calculate_anatomical_similarity(
                stable_features, 
                identity.get("stable_features", {})
            )
            
            if similarity >= self.ANATOMICAL_THRESHOLD and similarity > best_similarity:
                best_match_id = trace_id
                best_similarity = similarity
        
        if best_match_id and best_similarity >= self.ANATOMICAL_THRESHOLD:
            # Match found - update existing identity
            return self._update_identity(best_match_id, image, stable_features, transient_features)
        else:
            # No match found - return "not recognized" result
            return None, 0.0, {"status": "not_recognized", "best_similarity": best_similarity}
    
    def _create_new_identity(self, image: np.ndarray, stable_features: Dict, 
                           transient_features: Dict, name: Optional[str] = None, 
                           nickname: Optional[str] = None, analysis: Dict = None) -> Tuple[str, float, Dict]:
        """Create a new identity"""
        trace_id = self._generate_trace_id()
        
        # Save image
        image_path = self._save_image(image, trace_id, 1)
        
        # Create identity record
        identity = {
            "trace_id": trace_id,
            "stable_features": stable_features,
            "transient_features": transient_features,
            "image_references": [image_path],
            "recognition_metadata": {
                "last_seen": datetime.now().isoformat() + "Z",
                "confidence_score": 1.0,
                "matched_features": {
                    "anatomical": 1.0,
                    "accessories": 0.0,
                    "expression": 0.0
                }
            },
            "age_estimation": analysis.get("age_estimation", {}) if analysis else {},
            "name": name,
            "nickname": nickname,
            "observation_count": 1
        }
        
        self.identities[trace_id] = identity
        self._save_identities()
        
        return trace_id, 1.0, identity["recognition_metadata"]
    
    def _update_identity(self, trace_id: str, image: np.ndarray, 
                        stable_features: Dict, transient_features: Dict) -> Tuple[str, float, Dict]:
        """Update existing identity"""
        identity = self.identities[trace_id]
        
        # Update stable features (gradual adaptation)
        current_stable = identity.get("stable_features", {})
        updated_stable = self._update_moving_average(current_stable, stable_features, alpha=0.1)
        
        # Update transient features
        current_transient = identity.get("transient_features", {})
        updated_transient = self._update_moving_average(current_transient, transient_features, alpha=0.2)
        
        # Save new image with proper numbering
        existing_images = len(identity.get("image_references", []))
        scan_number = existing_images + 1
        image_path = self._save_image(image, trace_id, scan_number)
        
        # Update identity
        identity["stable_features"] = updated_stable
        identity["transient_features"] = updated_transient
        identity["image_references"].append(image_path)
        identity["recognition_metadata"]["last_seen"] = datetime.now().isoformat() + "Z"
        identity["observation_count"] = identity.get("observation_count", 0) + 1
        
        # Calculate confidence
        anatomical_sim = self._calculate_anatomical_similarity(updated_stable, stable_features)
        confidence = anatomical_sim * self.ANATOMICAL_WEIGHT
        
        identity["recognition_metadata"]["confidence_score"] = confidence
        identity["recognition_metadata"]["matched_features"] = {
            "anatomical": anatomical_sim,
            "accessories": 0.0,  # Simplified for now
            "expression": 0.0
        }
        
        self.identities[trace_id] = identity
        self._save_identities()
        
        return trace_id, confidence, identity["recognition_metadata"]
    
    def get_identity(self, trace_id: str) -> Optional[Dict]:
        """Get identity by trace ID"""
        return self.identities.get(trace_id)
    
    def list_identities(self) -> List[Dict]:
        """List all identities"""
        return list(self.identities.values())
    
    def search_by_name(self, name: str) -> Optional[Dict]:
        """Search for identity by name"""
        for identity in self.identities.values():
            if identity.get("name") == name:
                return identity
        return None
    
    def search_by_nickname(self, nickname: str) -> Optional[Dict]:
        """Search for identity by nickname"""
        for identity in self.identities.values():
            if identity.get("nickname") == nickname:
                return identity
        return None
    
    def search_by_name_or_nickname(self, query: str) -> Optional[Dict]:
        """Search for identity by name or nickname"""
        for identity in self.identities.values():
            if (identity.get("name") == query or 
                identity.get("nickname") == query):
                return identity
        return None
    
    def manual_rescan(self, trace_id: str, image: np.ndarray) -> Tuple[str, float, Dict]:
        """Manually rescan an existing identity"""
        if trace_id not in self.identities:
            raise ValueError(f"Identity {trace_id} not found")
        
        # Analyze face
        analysis = self._analyze_face_with_ai(image)
        
        # Extract features
        stable_features = {
            "face_numeric": analysis.get("face_numeric", {}),
            "shape_text": analysis.get("shape_text", []),
            "colors": analysis.get("colors", {})
        }
        
        # Force update with higher alpha for manual rescan
        identity = self.identities[trace_id]
        identity["stable_features"] = self._update_moving_average(
            identity.get("stable_features", {}), 
            stable_features, 
            alpha=0.5  # Higher update rate for manual rescan
        )
        
        # Save new reference image with proper numbering
        existing_images = len(identity.get("image_references", []))
        scan_number = existing_images + 1
        image_path = self._save_image(image, trace_id, scan_number)
        identity["image_references"].append(image_path)
        
        # Update metadata
        identity["recognition_metadata"]["last_seen"] = datetime.now().isoformat() + "Z"
        identity["observation_count"] = identity.get("observation_count", 0) + 1
        
        self._save_identities()
        
        return trace_id, 1.0, identity["recognition_metadata"]
    
    def delete_identity(self, trace_id: str) -> bool:
        """Delete an identity and its associated data"""
        if trace_id not in self.identities:
            return False
        
        # Remove image directory
        images_dir = self.config.get("images_dir", "images")
        image_dir = os.path.join(images_dir, trace_id)
        if os.path.exists(image_dir):
            import shutil
            shutil.rmtree(image_dir)
        
        # Remove from identities
        del self.identities[trace_id]
        self._save_identities()
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_identities = len(self.identities)
        total_observations = sum(identity.get("observation_count", 0) for identity in self.identities.values())
        
        return {
            "total_identities": total_identities,
            "total_observations": total_observations,
            "data_file": self.data_file,
            "next_id": self.next_id,
            "ai_provider": type(self.ai_provider).__name__,
            "ai_provider_available": self.ai_provider.is_available()
        }
    
    def switch_ai_provider(self, provider_type: str, **kwargs):
        """Switch to a different AI provider"""
        try:
            provider_config = self.config.get_ai_provider_config(provider_type)
            config = {**provider_config, **kwargs}
            self.ai_provider = AIProviderFactory.create_provider(provider_type, **config)
            if self.ai_provider.is_available():
                print(f"✅ Switched to {provider_type} AI provider")
            else:
                print(f"❌ {provider_type} provider is not available")
        except Exception as e:
            print(f"❌ Failed to switch to {provider_type} provider: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers"""
        return self.config.get_available_providers()
    
    def test_ai_provider(self) -> bool:
        """Test the current AI provider"""
        print(f"🧪 Testing {type(self.ai_provider).__name__} AI provider...")
        
        if not self.ai_provider.is_available():
            print("❌ AI provider is not available")
            return False
        
        # Create a test image (simple colored rectangle)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
        
        try:
            result = self.ai_provider.analyze_face(test_image)
            if result:
                print("✅ AI provider test successful")
                return True
            else:
                print("❌ AI provider test failed - no result")
                return False
        except Exception as e:
            print(f"❌ AI provider test failed: {e}")
            return False

# Dummy provider for when no AI providers are available
class DummyProvider(AIProvider):
    """Dummy provider that returns empty results"""
    
    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        return {}
    
    def compare_faces(self, test_image: np.ndarray, reference_data: Dict) -> float:
        return 0.0
    
    def is_available(self) -> bool:
        return False

# ===== UTILITY FUNCTIONS =====

def scan_myself(trace_system: TRACE, name: str = None, nickname: str = None):
    """Interactive self-registration (headless version)"""
    if not name:
        name = input("Enter your name: ").strip() or "Unknown"
    if not nickname:
        nickname = input("Enter your nickname (optional): ").strip() or None
    
    print(f"\n📸 Setting up camera for {name}...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return None
    
    print("✅ Camera ready!")
    print("\n📋 Instructions:")
    print("We'll take 3 photos: LEFT profile, RIGHT profile, and FRONT view")
    print("Make sure your face is well-lit for each pose")
    print("Type 'quit' to cancel at any time")
    print()
    
    poses = ["LEFT profile", "RIGHT profile", "FRONT view"]
    captured_images = []
    
    print("\n📸 We'll capture all 3 photos first, then analyze them together...")
    
    # Capture all 3 photos first
    for scan_count in range(3):
        pose = poses[scan_count]
        print(f"\n📸 Capture {scan_count + 1}/3: {pose}")
        print(f"Turn your head to the {pose.lower()} and press ENTER when ready...")
        
        user_input = input("Press ENTER to capture (or 'quit' to cancel): ").strip().lower()
        
        if user_input == 'quit':
            print("❌ Registration cancelled")
            break
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading from camera")
            break
        
        print("📸 Capturing image...")
        time.sleep(0.5)
        
        # Store the captured image
        captured_images.append((frame, pose))
        print(f"✅ {pose} captured!")
    
    # Now process all images together
    trace_id = None
    if captured_images:
        print(f"\n🤖 Processing {len(captured_images)} images with AI analysis...")
        print("This may take a moment...")
        
        try:
            # Process first image to create identity
            first_frame, first_pose = captured_images[0]
            trace_id, confidence, metadata = trace_system.scan_person(first_frame, name=name, nickname=nickname)
            print(f"✅ Created identity: {trace_id}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Pose: {first_pose}")
            
            # Process remaining images
            for i in range(1, len(captured_images)):
                frame, pose = captured_images[i]
                print(f"   Processing {pose}...")
                try:
                    trace_id, confidence, metadata = trace_system.manual_rescan(trace_id, frame)
                    print(f"✅ Updated identity: {trace_id}")
                    print(f"   Confidence: {confidence:.2f}")
                    print(f"   Pose: {pose}")
                except Exception as e:
                    print(f"❌ Error processing {pose}: {e}")
                    print(f"   Continuing with next image...")
                    continue
            
            print(f"\n🎉 All {len(captured_images)} poses processed!")
            
        except Exception as e:
            print(f"❌ Error during AI processing: {e}")
            print("   Please try again")
            return None
    
    cap.release()
    
    if trace_id is not None:
        print(f"\n🎉 Successfully registered {name}!")
        if nickname:
            print(f"   Name: {name} ({nickname})")
        else:
            print(f"   Name: {name}")
        print(f"📁 Data saved to: {trace_system.data_file}")
        print(f"🖼️  Images saved to: {trace_system.config.get('images_dir', 'images')}/{trace_id}/")
        
        # Show final statistics
        stats = trace_system.get_statistics()
        print(f"\n📊 TRACE Statistics:")
        print(f"   Total identities: {stats['total_identities']}")
        print(f"   Total observations: {stats['total_observations']}")
        print(f"   AI Provider: {stats['ai_provider']}")
        
        return trace_id
    else:
        print("❌ No scans completed")
        return None

def view_identities(trace_system: TRACE):
    """View all identities in the system"""
    print("=== TRACE Identities ===")
    
    identities = trace_system.list_identities()
    
    if not identities:
        print("❌ No identities found in TRACE system")
        return
    
    print(f"📊 Found {len(identities)} identity(ies) in TRACE system")
    print()
    
    for i, identity in enumerate(identities, 1):
        print(f"👤 Identity #{i}:")
        print(f"   ID: {identity['trace_id']}")
        print(f"   Name: {identity.get('name', 'Unknown')}")
        if identity.get('nickname'):
            print(f"   Nickname: {identity.get('nickname')}")
        print(f"   Images: {len(identity.get('image_references', []))}")
        print(f"   Last seen: {identity['recognition_metadata']['last_seen']}")
        print(f"   Confidence: {identity['recognition_metadata']['confidence_score']:.2f}")
        print()
        
        # Show anatomical features
        stable_features = identity.get('stable_features', {})
        face_numeric = stable_features.get('face_numeric', {})
        if face_numeric:
            print("📏 Anatomical Measurements:")
            for key, value in list(face_numeric.items())[:10]:  # Show first 10
                print(f"   {key}: {value}")
            print()
        
        # Show colors
        colors = stable_features.get('colors', {})
        if colors:
            print("🎨 Color Analysis:")
            for color_type, rgb in colors.items():
                print(f"   {color_type}: RGB{rgb}")
            print()
        
        # Show age estimation
        age_info = identity.get('age_estimation', {})
        if age_info:
            print("👤 Age Analysis:")
            print(f"   Estimated Age: {age_info.get('estimated_age', 'Unknown')}")
            print(f"   Age Range: {age_info.get('age_range', 'Unknown')}")
            print(f"   Confidence: {age_info.get('age_confidence', 0):.2f}")
            
            # Show detailed analysis
            detailed = age_info.get('detailed_analysis', {})
            if detailed:
                print("   Detailed Analysis:")
                for key, value in detailed.items():
                    print(f"     {key.replace('_', ' ').title()}: {value}")
            
            indicators = age_info.get('age_indicators', [])
            if indicators:
                print(f"   Indicators: {', '.join(indicators)}")
            print()
    
    # Show file locations
    print("📁 File Locations:")
    print(f"   JSON: {os.path.abspath(trace_system.data_file)}")
    print(f"   Images: {os.path.abspath(trace_system.config.get('images_dir', 'images'))}")
    
    # Show statistics
    stats = trace_system.get_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Total identities: {stats['total_identities']}")
    print(f"   Total observations: {stats['total_observations']}")
    print(f"   AI Provider: {stats['ai_provider']}")
    print(f"   AI Provider Available: {stats['ai_provider_available']}")

def update_name(trace_system: TRACE, trace_id: str, new_name: str = None, new_nickname: str = None):
    """Update name and nickname for an identity"""
    if trace_id not in trace_system.identities:
        print(f"❌ Identity {trace_id} not found")
        return False
    
    identity = trace_system.identities[trace_id]
    
    if new_name is None:
        new_name = input(f"Enter new name (current: {identity.get('name', 'Unknown')}): ").strip()
    if new_nickname is None:
        current_nickname = identity.get('nickname')
        new_nickname = input(f"Enter new nickname (current: {current_nickname or 'None'}): ").strip() or None
    
    identity['name'] = new_name
    identity['nickname'] = new_nickname
    
    trace_system._save_identities()
    
    print(f"✅ Updated {trace_id}:")
    print(f"   Name: {new_name}")
    if new_nickname:
        print(f"   Nickname: {new_nickname}")
    
    return True

def person_test(trace_system: TRACE):
    """Test person recognition with 85% similarity threshold"""
    print("\n🧪 Person Test - 85% Similarity Check")
    print("=" * 50)
    
    # Check if we have any identities
    if not trace_system.identities:
        print("❌ No identities found in database")
        print("   Please scan yourself first (option 1)")
        return False
    
    print("📸 Capturing test image...")
    print("Position your face in the camera and press ENTER when ready...")
    
    user_input = input("Press ENTER to capture (or 'quit' to cancel): ").strip().lower()
    if user_input == 'quit':
        print("❌ Test cancelled")
        return False
    
    # Capture image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not capture image")
        return False
    
    print("🤖 Analyzing face...")
    
    # Check if AI provider is available
    if not trace_system.ai_provider.is_available():
        print("❌ Error: AI provider is not available")
        print("   Please configure an AI provider")
        return False
    
    try:
        # Save the test image temporarily
        test_image_path = "temp_test_image.jpg"
        cv2.imwrite(test_image_path, frame)
        
        print("🔍 AI comparing test image against stored identities...")
        
        # Find best match using AI comparison
        best_match_id = None
        best_similarity = 0.0
        
        for trace_id, identity in trace_system.identities.items():
            print(f"   Comparing with {trace_id}...")
            
            # Get stored images and JSON data
            image_references = identity.get("image_references", [])
            stored_features = identity.get("stable_features", {})
            
            if not image_references and not stored_features:
                print(f"     No data available for {trace_id}")
                continue
            
            # Use AI to compare test image with stored data
            similarity = trace_system.ai_provider.compare_faces(frame, stored_features)
            print(f"   {trace_id}: {similarity:.3f} ({similarity*100:.1f}%)")
            
            if similarity > best_similarity:
                best_match_id = trace_id
                best_similarity = similarity
        
        # Clean up temp file
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        # Check if match meets threshold
        print(f"\n📊 Results:")
        print(f"   Best Match: {best_match_id or 'None'}")
        print(f"   Similarity: {best_similarity:.3f} ({best_similarity*100:.1f}%)")
        print(f"   Threshold: 0.850 (85.0%)")
        
        if best_similarity >= 0.85:
            identity = trace_system.identities[best_match_id]
            name = identity.get('name', 'Unknown')
            nickname = identity.get('nickname')
            
            print(f"\n✅ ACCESS GRANTED!")
            print(f"   Identity: {best_match_id}")
            print(f"   Name: {name}")
            if nickname:
                print(f"   Nickname: {nickname}")
            print(f"   Confidence: {best_similarity*100:.1f}%")
            
            # Update last seen
            identity['recognition_metadata']['last_seen'] = datetime.now().isoformat() + "Z"
            identity['recognition_metadata']['confidence_score'] = best_similarity
            trace_system._save_identities()
            
            return True
        else:
            print(f"\n❌ ACCESS DENIED!")
            print(f"   Similarity {best_similarity*100:.1f}% is below 85% threshold")
            print(f"   Required: 85.0% or higher")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if "API is required" in str(e):
            print("   Please check your configuration and ensure API keys are set correctly")
        elif "quota" in str(e).lower():
            print("   API quota exceeded. Please wait or upgrade your plan")
        elif "404" in str(e):
            print("   API model not found. Please check the model name")
        return False

def test_camera():
    """Test camera functionality"""
    print("=== Camera Test ===")
    print("Testing camera functionality...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    print("✅ Camera opened successfully!")
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read from camera")
        cap.release()
        return False
    
    print("✅ Successfully read frame from camera")
    print(f"   Frame size: {frame.shape}")
    
    test_image_path = "camera_test.jpg"
    cv2.imwrite(test_image_path, frame)
    print(f"✅ Saved test image: {test_image_path}")
    
    cap.release()
    
    if os.path.exists(test_image_path):
        print(f"✅ Test image file exists: {os.path.abspath(test_image_path)}")
        os.remove(test_image_path)
        print("✅ Camera test completed successfully!")
        return True
    else:
        print("❌ Test image file not found")
        return False

# ===== MAIN EXECUTION =====

def main():
    """Main function for direct execution"""
    print("=== TRACE Multi-AI System ===")
    print("Supporting: Gemini, OpenAI, Claude, and Local LLMs")
    print()
    
    # Initialize TRACE system
    trace = TRACE()
    
    # Show configuration
    print("🔧 Configuration:")
    print(f"   AI Provider: {type(trace.ai_provider).__name__}")
    print(f"   Available Providers: {', '.join(trace.get_available_providers())}")
    print(f"   Data File: {trace.data_file}")
    print()
    
    # Test AI provider
    if not trace.test_ai_provider():
        print("❌ AI provider test failed. Please check your configuration.")
        print("   Available providers:", ', '.join(trace.get_available_providers()))
        return
    
    print("1. Scan yourself")
    print("2. View identities")
    print("3. Test camera")
    print("4. Update name")
    print("5. Person Test (85% similarity check)")
    print("6. Switch AI provider")
    print("7. Test AI provider")
    print("8. Exit")
    
    while True:
        choice = input("\nChoose option (1-8): ").strip()
        
        if choice == '1':
            scan_myself(trace)
        elif choice == '2':
            view_identities(trace)
        elif choice == '3':
            test_camera()
        elif choice == '4':
            trace_id = input("Enter trace ID: ").strip()
            update_name(trace, trace_id)
        elif choice == '5':
            person_test(trace)
        elif choice == '6':
            print("Available providers:", ', '.join(trace.get_available_providers()))
            provider = input("Enter provider name: ").strip()
            if provider in trace.get_available_providers():
                trace.switch_ai_provider(provider)
            else:
                print("❌ Provider not available")
        elif choice == '7':
            trace.test_ai_provider()
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
