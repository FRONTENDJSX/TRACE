"""
AI Provider Abstraction Layer for TRACE
Supports multiple AI APIs and local LLMs
"""

import json
import base64
import io
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import numpy as np
import cv2

# Base AI Provider Interface
class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face and return comprehensive features"""
        pass
    
    @abstractmethod
    def compare_faces(self, test_image: np.ndarray, reference_data: Dict) -> float:
        """Compare test image with reference data and return similarity score"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        pass

# Google Gemini Provider
class GeminiProvider(AIProvider):
    """Google Gemini API provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face using Gemini API"""
        if not self.is_available():
            return self._fallback_analysis()
        
        try:
            # Convert OpenCV image to PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            prompt = self._get_analysis_prompt()
            response = self.model.generate_content([prompt, pil_image])
            
            # Parse JSON response
            response_text = response.text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._fallback_analysis()
    
    def compare_faces(self, test_image: np.ndarray, reference_data: Dict) -> float:
        """Compare faces using Gemini API"""
        if not self.is_available():
            return 0.0
        
        try:
            # Convert test image
            image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Prepare reference data description
            json_description = self._create_json_description(reference_data)
            
            prompt = f"""
            Compare this test image with the stored identity data.
            
            STORED IDENTITY DATA:
            {json_description}
            
            TASK: Determine the similarity between the test image and the stored identity.
            Consider facial structure, bone features, eye shape, nose shape, jaw structure,
            facial proportions, and overall facial geometry.
            
            Return ONLY a JSON object with this exact format:
            {{
                "similarity_score": 0.85,
                "confidence": 0.92,
                "reasoning": "Detailed explanation of the comparison"
            }}
            
            Similarity score should be between 0.0 (completely different) and 1.0 (identical).
            """
            
            response = self.model.generate_content([prompt, pil_image])
            response_text = response.text
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return float(result.get("similarity_score", 0.0))
            else:
                return 0.0
                
        except Exception as e:
            print(f"Gemini comparison error: {e}")
            return 0.0
    
    def _get_analysis_prompt(self) -> str:
        """Get the face analysis prompt for Gemini"""
        return """
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
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when API is not available"""
        print("❌ Gemini API is required for face analysis. Please provide a valid GEMINI_API_KEY.")
        return {}

# OpenAI Provider
class OpenAIProvider(AIProvider):
    """OpenAI GPT-4 Vision API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.api_key = api_key
        self.model_name = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize OpenAI API"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face using OpenAI GPT-4 Vision"""
        if not self.is_available():
            return self._fallback_analysis()
        
        try:
            # Convert image to base64
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            prompt = self._get_analysis_prompt()
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._fallback_analysis()
    
    def compare_faces(self, test_image: np.ndarray, reference_data: Dict) -> float:
        """Compare faces using OpenAI GPT-4 Vision"""
        if not self.is_available():
            return 0.0
        
        try:
            # Convert test image to base64
            image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare reference data description
            json_description = self._create_json_description(reference_data)
            
            prompt = f"""
            Compare this test image with the stored identity data.
            
            STORED IDENTITY DATA:
            {json_description}
            
            TASK: Determine the similarity between the test image and the stored identity.
            Consider facial structure, bone features, eye shape, nose shape, jaw structure,
            facial proportions, and overall facial geometry.
            
            Return ONLY a JSON object with this exact format:
            {{
                "similarity_score": 0.85,
                "confidence": 0.92,
                "reasoning": "Detailed explanation of the comparison"
            }}
            
            Similarity score should be between 0.0 (completely different) and 1.0 (identical).
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return float(result.get("similarity_score", 0.0))
            else:
                return 0.0
                
        except Exception as e:
            print(f"OpenAI comparison error: {e}")
            return 0.0
    
    def _get_analysis_prompt(self) -> str:
        """Get the face analysis prompt for OpenAI"""
        return """
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
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when API is not available"""
        print("❌ OpenAI API is required for face analysis. Please provide a valid OPENAI_API_KEY.")
        return {}

# Anthropic Claude Provider
class ClaudeProvider(AIProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model_name = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Anthropic API"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as e:
            print(f"Failed to initialize Anthropic: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face using Anthropic Claude"""
        if not self.is_available():
            return self._fallback_analysis()
        
        try:
            # Convert image to base64
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            prompt = self._get_analysis_prompt()
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_str
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"Claude API error: {e}")
            return self._fallback_analysis()
    
    def compare_faces(self, test_image: np.ndarray, reference_data: Dict) -> float:
        """Compare faces using Anthropic Claude"""
        if not self.is_available():
            return 0.0
        
        try:
            # Convert test image to base64
            image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare reference data description
            json_description = self._create_json_description(reference_data)
            
            prompt = f"""
            Compare this test image with the stored identity data.
            
            STORED IDENTITY DATA:
            {json_description}
            
            TASK: Determine the similarity between the test image and the stored identity.
            Consider facial structure, bone features, eye shape, nose shape, jaw structure,
            facial proportions, and overall facial geometry.
            
            Return ONLY a JSON object with this exact format:
            {{
                "similarity_score": 0.85,
                "confidence": 0.92,
                "reasoning": "Detailed explanation of the comparison"
            }}
            
            Similarity score should be between 0.0 (completely different) and 1.0 (identical).
            """
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_str
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                return float(result.get("similarity_score", 0.0))
            else:
                return 0.0
                
        except Exception as e:
            print(f"Claude comparison error: {e}")
            return 0.0
    
    def _get_analysis_prompt(self) -> str:
        """Get the face analysis prompt for Claude"""
        return """
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
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when API is not available"""
        print("❌ Anthropic Claude API is required for face analysis. Please provide a valid ANTHROPIC_API_KEY.")
        return {}

# Local LLM Provider (Ollama)
class OllamaProvider(AIProvider):
    """Local LLM provider using Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llava:7b"):
        self.base_url = base_url
        self.model_name = model
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Ollama client"""
        try:
            import requests
            self.client = requests
            # Test connection
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {self.base_url}")
            else:
                print(f"❌ Failed to connect to Ollama at {self.base_url}")
                self.client = None
        except Exception as e:
            print(f"Failed to initialize Ollama: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze face using local Ollama model"""
        if not self.is_available():
            return self._fallback_analysis()
        
        try:
            # Convert image to base64
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            prompt = self._get_analysis_prompt()
            
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_str],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Parse JSON response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    return self._fallback_analysis()
            else:
                print(f"Ollama API error: {response.status_code}")
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"Ollama API error: {e}")
            return self._fallback_analysis()
    
    def compare_faces(self, test_image: np.ndarray, reference_data: Dict) -> float:
        """Compare faces using local Ollama model"""
        if not self.is_available():
            return 0.0
        
        try:
            # Convert test image to base64
            image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Prepare reference data description
            json_description = self._create_json_description(reference_data)
            
            prompt = f"""
            Compare this test image with the stored identity data.
            
            STORED IDENTITY DATA:
            {json_description}
            
            TASK: Determine the similarity between the test image and the stored identity.
            Consider facial structure, bone features, eye shape, nose shape, jaw structure,
            facial proportions, and overall facial geometry.
            
            Return ONLY a JSON object with this exact format:
            {{
                "similarity_score": 0.85,
                "confidence": 0.92,
                "reasoning": "Detailed explanation of the comparison"
            }}
            
            Similarity score should be between 0.0 (completely different) and 1.0 (identical).
            """
            
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_str],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Parse JSON response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return float(result.get("similarity_score", 0.0))
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Ollama comparison error: {e}")
            return 0.0
    
    def _get_analysis_prompt(self) -> str:
        """Get the face analysis prompt for Ollama"""
        return """
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
    
    def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when API is not available"""
        print("❌ Ollama is required for face analysis. Please ensure Ollama is running and the model is available.")
        return {}

# AI Provider Factory
class AIProviderFactory:
    """Factory for creating AI providers"""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> AIProvider:
        """Create an AI provider based on type"""
        provider_type = provider_type.lower()
        
        if provider_type == "gemini":
            return GeminiProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gemini-2.5-flash")
            )
        elif provider_type == "openai":
            return OpenAIProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gpt-4-vision-preview")
            )
        elif provider_type == "claude":
            return ClaudeProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "claude-3-5-sonnet-20241022")
            )
        elif provider_type == "ollama":
            return OllamaProvider(
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                model=kwargs.get("model", "llava:7b")
            )
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available provider types"""
        return ["gemini", "openai", "claude", "ollama"]
