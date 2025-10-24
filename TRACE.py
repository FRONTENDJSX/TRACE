"""
TRACE - Technical Recognition and Anatomical Character Engine
Dynamic human identity recognition system with anatomical-based recognition

CONSOLIDATED VERSION - All TRACE functionality in one file
"""

import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
import base64
import io

class TRACE:
    def __init__(self, data_file: str = "TRACE.json", gemini_api_key: str = None):
        """
        Initialize TRACE system
        
        Args:
            data_file: Path to TRACE.json file
            gemini_api_key: Google Gemini API key for face analysis
        """
        self.data_file = data_file
        self.next_id = 1
        
        # Create images directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        # Initialize Gemini API
        self.gemini_api_key = gemini_api_key
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
            print("Warning: No Gemini API key provided. Face analysis will be limited.")
        
        # Load existing identities
        self.identities = self._load_identities()
        self._update_next_id()
        
        # Recognition thresholds
        self.ANATOMICAL_THRESHOLD = 0.85
        self.ANATOMICAL_WEIGHT = 0.95
        self.ACCESSORY_WEIGHT = 0.02
        self.EXPRESSION_WEIGHT = 0.0
    
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
        image_dir = os.path.join("images", trace_id)
        os.makedirs(image_dir, exist_ok=True)
        
        filename = f"scan_{scan_number:03d}.jpg"
        image_path = os.path.join(image_dir, filename)
        
        cv2.imwrite(image_path, image)
        return image_path
    
    def _analyze_face_with_gemini(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face using Gemini API"""
        if not self.model:
            return self._fallback_analysis()
        
        print(f"🤖 Calling Gemini API with image shape: {image.shape}")
        try:
            # Convert OpenCV image to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64 for API
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            prompt = """
            Analyze this face image and provide comprehensive anatomical measurements and features for facial recognition.
            Return a JSON object with the following structure containing 60+ facial features:
            {
                "face_numeric": {
                    "face_height_in": float,
                    "face_width_in": float,
                    "face_length_in": float,
                    "jaw_length_in": float,
                    "jaw_width_in": float,
                    "jaw_angle_deg": float,
                    "jaw_angle_left_deg": float,
                    "jaw_angle_right_deg": float,
                    "cheekbone_width_in": float,
                    "cheekbone_height_in": float,
                    "cheekbone_prominence": float,
                    "interocular_distance_in": float,
                    "eye_width_left_in": float,
                    "eye_width_right_in": float,
                    "eye_height_left_in": float,
                    "eye_height_right_in": float,
                    "eye_depth_left_in": float,
                    "eye_depth_right_in": float,
                    "eye_angle_left_deg": float,
                    "eye_angle_right_deg": float,
                    "nose_length_in": float,
                    "nose_width_in": float,
                    "nose_height_in": float,
                    "nose_bridge_width_in": float,
                    "nose_tip_width_in": float,
                    "nose_angle_deg": float,
                    "nostril_width_in": float,
                    "nostril_height_in": float,
                    "lip_width_in": float,
                    "lip_height_in": float,
                    "lip_thickness_upper_in": float,
                    "lip_thickness_lower_in": float,
                    "lip_angle_deg": float,
                    "lip_curve_deg": float,
                    "forehead_width_in": float,
                    "forehead_height_in": float,
                    "forehead_angle_deg": float,
                    "temple_width_in": float,
                    "temple_depth_in": float,
                    "chin_width_in": float,
                    "chin_height_in": float,
                    "chin_angle_deg": float,
                    "chin_prominence": float,
                    "ear_width_left_in": float,
                    "ear_width_right_in": float,
                    "ear_height_left_in": float,
                    "ear_height_right_in": float,
                    "ear_angle_left_deg": float,
                    "ear_angle_right_deg": float,
                    "eyebrow_width_left_in": float,
                    "eyebrow_width_right_in": float,
                    "eyebrow_height_left_in": float,
                    "eyebrow_height_right_in": float,
                    "eyebrow_angle_left_deg": float,
                    "eyebrow_angle_right_deg": float,
                    "eyebrow_thickness_left": float,
                    "eyebrow_thickness_right": float,
                    "eyebrow_arch_left_deg": float,
                    "eyebrow_arch_right_deg": float,
                    "facial_symmetry_score": float,
                    "face_oval_ratio": float,
                    "face_triangle_ratio": float,
                    "face_square_ratio": float,
                    "face_heart_ratio": float,
                    "face_diamond_ratio": float,
                    "face_round_ratio": float
                },
                "shape_text": [
                    "jaw_contour: detailed description of jawline shape and angles",
                    "eye_contour: detailed description of eye shape and positioning",
                    "nose_contour: detailed description of nose shape and bridge",
                    "lip_contour: detailed description of lip shape and fullness",
                    "forehead_contour: detailed description of forehead shape",
                    "cheek_contour: detailed description of cheekbone structure",
                    "chin_contour: detailed description of chin shape and prominence",
                    "ear_contour: detailed description of ear shape and positioning",
                    "eyebrow_contour: detailed description of eyebrow shape and arch"
                ],
                "colors": {
                    "hair_rgb": [r, g, b],
                    "eye_rgb": [r, g, b],
                    "skin_rgb": [r, g, b],
                    "eyebrow_rgb": [r, g, b],
                    "lip_rgb": [r, g, b]
                },
                "expression": "neutral|smile|frown|surprise|anger|sad|disgust|fear",
                "age_estimation": {
                    "estimated_age": int,
                    "age_confidence": float,
                    "age_range": "string (e.g., '12-14', '15-17', '18-20', '21-25')",
                    "age_indicators": ["facial_bone_structure", "skin_texture", "eye_area", "jaw_development", "cheekbone_prominence", "facial_maturity", "youth_indicators"],
                    "detailed_analysis": {
                        "facial_bone_development": "string (underdeveloped/developing/mature)",
                        "skin_quality": "string (youthful/maturing/aged)",
                        "eye_area": "string (youthful/maturing/mature)",
                        "jaw_structure": "string (soft/developing/defined)",
                        "overall_maturity": "string (child/teen/young_adult/adult)"
                    }
                },
                "accessories": {
                    "glasses": {"present": bool, "description": "string", "type": "string"},
                    "jewelry": {"present": bool, "description": "string", "type": "string"},
                    "facial_hair": {"present": bool, "description": "string", "type": "string"},
                    "hat": {"present": bool, "description": "string", "type": "string"}
                }
            }
            """
            
            response = self.model.generate_content([prompt, pil_image])
            
            # Parse JSON response
            response_text = response.text
            print(f"🤖 Gemini API response length: {len(response_text)}")
            print(f"🤖 First 200 chars: {response_text[:200]}...")
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                face_numeric = result.get('face_numeric', {})
                print(f"🤖 Parsed JSON with {len(face_numeric)} numeric features")
                
                # Check if face was detected (all -1.0 values means no face detected)
                sample_values = list(face_numeric.values())[:5]
                if all(v == -1.0 for v in sample_values):
                    print("🤖 ⚠️ No face detected in image - all values are -1.0")
                else:
                    print(f"🤖 ✅ Face detected - sample values: {sample_values}")
                
                return result
            else:
                print("🤖 No JSON found in response, using fallback")
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when Gemini API is not available - NO DUMMY DATA"""
        print("❌ Gemini API is required for face analysis. Please provide a valid GEMINI_API_KEY in your .env file.")
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
        analysis = self._analyze_face_with_gemini(image)
        
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
        analysis = self._analyze_face_with_gemini(image)
        
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
        analysis = self._analyze_face_with_gemini(image)
        
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
        image_dir = os.path.join("images", trace_id)
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
            "next_id": self.next_id
        }
    
    # ===== UTILITY FUNCTIONS =====
    
    def scan_myself(self, name: str = None, nickname: str = None):
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
        if captured_images:
            print(f"\n🤖 Processing {len(captured_images)} images with AI analysis...")
            print("This may take a moment...")
            
            try:
                # Process first image to create identity
                first_frame, first_pose = captured_images[0]
                trace_id, confidence, metadata = self.scan_person(first_frame, name=name, nickname=nickname)
                print(f"✅ Created identity: {trace_id}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Pose: {first_pose}")
                
                # Process remaining images
                for i in range(1, len(captured_images)):
                    frame, pose = captured_images[i]
                    print(f"   Processing {pose}...")
                    try:
                        trace_id, confidence, metadata = self.manual_rescan(trace_id, frame)
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
            print(f"📁 Data saved to: {self.data_file}")
            print(f"🖼️  Images saved to: images/{trace_id}/")
            
            # Show final statistics
            stats = self.get_statistics()
            print(f"\n📊 TRACE Statistics:")
            print(f"   Total identities: {stats['total_identities']}")
            print(f"   Total observations: {stats['total_observations']}")
            
            return trace_id
        else:
            print("❌ No scans completed")
            return None
    
    def view_identities(self):
        """View all identities in the system"""
        print("=== TRACE Identities ===")
        
        identities = self.list_identities()
        
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
                for key, value in face_numeric.items():
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
        print(f"   JSON: {os.path.abspath(self.data_file)}")
        print(f"   Images: {os.path.abspath('images')}")
        
        # Show statistics
        stats = self.get_statistics()
        print(f"\n📊 System Statistics:")
        print(f"   Total identities: {stats['total_identities']}")
        print(f"   Total observations: {stats['total_observations']}")
    
    def update_name(self, trace_id: str, new_name: str = None, new_nickname: str = None):
        """Update name and nickname for an identity"""
        if trace_id not in self.identities:
            print(f"❌ Identity {trace_id} not found")
            return False
        
        identity = self.identities[trace_id]
        
        if new_name is None:
            new_name = input(f"Enter new name (current: {identity.get('name', 'Unknown')}): ").strip()
        if new_nickname is None:
            current_nickname = identity.get('nickname')
            new_nickname = input(f"Enter new nickname (current: {current_nickname or 'None'}): ").strip() or None
        
        identity['name'] = new_name
        identity['nickname'] = new_nickname
        
        self._save_identities()
        
        print(f"✅ Updated {trace_id}:")
        print(f"   Name: {new_name}")
        if new_nickname:
            print(f"   Nickname: {new_nickname}")
        
        return True
    
    def _ai_compare_images(self, test_image_path: str, trace_id: str, identity: Dict) -> float:
        """Use AI to compare test image with stored identity data"""
        try:
            # Load test image
            test_image = Image.open(test_image_path)
            
            # Prepare stored images
            stored_images = []
            image_references = identity.get("image_references", [])
            
            for img_path in image_references:
                if os.path.exists(img_path):
                    stored_images.append(Image.open(img_path))
            
            # Get JSON description
            stored_features = identity.get("stable_features", {})
            json_description = self._create_json_description(stored_features)
            
            # Create comparison prompt
            prompt = f"""
            Compare this test image with the stored identity data for {trace_id}.
            
            STORED IDENTITY DATA:
            {json_description}
            
            TASK: Determine the similarity between the test image and the stored identity.
            Consider:
            1. Facial structure and bone features
            2. Eye shape, nose shape, jaw structure
            3. Facial proportions and ratios
            4. Overall facial geometry
            5. Distinctive features mentioned in the JSON data
            
            Return ONLY a JSON object with this exact format:
            {{
                "similarity_score": 0.85,
                "confidence": 0.92,
                "reasoning": "Detailed explanation of the comparison"
            }}
            
            Similarity score should be between 0.0 (completely different) and 1.0 (identical).
            """
            
            # Prepare images for comparison
            images_to_send = [test_image] + stored_images[:2]  # Limit to 3 images max
            
            # Call Gemini API
            response = self.model.generate_content([prompt] + images_to_send)
            response_text = response.text
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return float(result.get("similarity_score", 0.0))
            else:
                print(f"     Could not parse AI response for {trace_id}")
                return 0.0
                
        except Exception as e:
            print(f"     Error comparing {trace_id}: {e}")
            return 0.0
    
    def _create_json_description(self, stored_features: Dict) -> str:
        """Create a readable description from stored features"""
        if not stored_features:
            return "No stored features available"
        
        description = "STORED IDENTITY FEATURES:\n"
        
        # Add numeric measurements
        face_numeric = stored_features.get("face_numeric", {})
        if face_numeric:
            description += "\nFACIAL MEASUREMENTS:\n"
            for key, value in list(face_numeric.items())[:10]:  # Show first 10 measurements
                description += f"  {key}: {value}\n"
        
        # Add shape descriptions
        shape_text = stored_features.get("shape_text", [])
        if shape_text:
            description += "\nFACIAL SHAPE DESCRIPTIONS:\n"
            for shape in shape_text:
                description += f"  {shape}\n"
        
        # Add colors
        colors = stored_features.get("colors", {})
        if colors:
            description += "\nCOLOR ANALYSIS:\n"
            for color_type, rgb in colors.items():
                description += f"  {color_type}: RGB{rgb}\n"
        
        return description
    
    def person_test(self):
        """Test person recognition with 85% similarity threshold"""
        print("\n🧪 Person Test - 85% Similarity Check")
        print("=" * 50)
        
        # Check if we have any identities
        if not self.identities:
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
        
        # Check if API key is available
        if not self.gemini_api_key:
            print("❌ Error: No Gemini API key found")
            print("   Please set GEMINI_API_KEY in your .env file")
            return False
        
        try:
            # Save the test image temporarily
            test_image_path = "temp_test_image.jpg"
            cv2.imwrite(test_image_path, frame)
            
            print("🔍 AI comparing test image against stored identities...")
            
            # Find best match using AI comparison
            best_match_id = None
            best_similarity = 0.0
            
            for trace_id, identity in self.identities.items():
                print(f"   Comparing with {trace_id}...")
                
                # Get stored images and JSON data
                image_references = identity.get("image_references", [])
                stored_features = identity.get("stable_features", {})
                
                if not image_references and not stored_features:
                    print(f"     No data available for {trace_id}")
                    continue
                
                # Use AI to compare test image with stored data
                similarity = self._ai_compare_images(test_image_path, trace_id, identity)
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
                identity = self.identities[best_match_id]
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
                self._save_identities()
                
                return True
            else:
                print(f"\n❌ ACCESS DENIED!")
                print(f"   Similarity {best_similarity*100:.1f}% is below 85% threshold")
                print(f"   Required: 85.0% or higher")
                return False
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            if "Gemini API is required" in str(e):
                print("   Please check your .env file and ensure GEMINI_API_KEY is set correctly")
            elif "quota" in str(e).lower():
                print("   API quota exceeded. Please wait or upgrade your plan")
            elif "404" in str(e):
                print("   API model not found. Please check the model name")
            return False
    
    def test_camera(self):
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
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    trace = TRACE(gemini_api_key=gemini_key)
    
    print("=== TRACE System ===")
    print("1. Scan yourself")
    print("2. View identities")
    print("3. Test camera")
    print("4. Update name")
    print("5. Person Test (85% similarity check)")
    print("6. Exit")
    
    while True:
        choice = input("\nChoose option (1-6): ").strip()
        
        if choice == '1':
            trace.scan_myself()
        elif choice == '2':
            trace.view_identities()
        elif choice == '3':
            trace.test_camera()
        elif choice == '4':
            trace_id = input("Enter trace ID: ").strip()
            trace.update_name(trace_id)
        elif choice == '5':
            trace.person_test()
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()