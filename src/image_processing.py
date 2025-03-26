"""
Image Processing Module for Medical Report Analysis

This module handles the processing of medical report images to extract relevant
information for heart disease prediction.
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Dict, List, Tuple, Optional
import re

def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a medical report image.

    Args:
        image_path (str): Path to the image file

    Returns:
        np.ndarray: Preprocessed image array
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return gray

def extract_text_from_image(image: np.ndarray) -> str:
    """
    Extract text from the preprocessed image using OCR.

    Args:
        image (np.ndarray): Preprocessed image array

    Returns:
        str: Extracted text from the image
    """
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text

def parse_medical_report(text: str) -> Dict[str, float]:
    """
    Parse the extracted text to find relevant medical measurements.
    Extracts all values matching the Cleveland dataset format.

    Args:
        text (str): Extracted text from the medical report

    Returns:
        Dict[str, float]: Dictionary containing extracted medical measurements
    """
    measurements = {}
    
    # Define patterns for different measurements
    patterns = {
        'age': r'(?:age|patient age)[:\s]+(\d+)',
        'sex': r'(?:sex|gender)[:\s]+(?:male|M|1)[:\s]*(\d+)',
        'cp': r'(?:chest pain|CP)[:\s]+(?:type|grade)[:\s]*(\d+)',
        'trestbps': r'(?:blood pressure|BP|resting BP)[:\s]+(\d+)',
        'chol': r'(?:cholesterol|chol)[:\s]+(\d+)',
        'fbs': r'(?:fasting blood sugar|FBS)[:\s]+(\d+)',
        'restecg': r'(?:resting ECG|ECG)[:\s]+(?:result|finding)[:\s]*(\d+)',
        'thalach': r'(?:max heart rate|MHR|thalach)[:\s]+(\d+)',
        'exang': r'(?:exercise angina|exang)[:\s]+(?:yes|1|no|0)[:\s]*(\d+)',
        'oldpeak': r'(?:ST depression|oldpeak)[:\s]+([\d.]+)',
        'slope': r'(?:ST slope|slope)[:\s]+(\d+)',
        'ca': r'(?:vessels|vessel count|ca)[:\s]+(\d+)',
        'thal': r'(?:thalassemia|thal)[:\s]+(\d+)'
    }
    
    # Extract measurements using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text.lower())
        if match:
            try:
                value = float(match.group(1))
                # Validate the extracted value based on known ranges
                if key == 'sex' and value not in [0, 1]:
                    continue
                elif key == 'cp' and value not in [1, 2, 3, 4]:
                    continue
                elif key == 'fbs' and value not in [0, 1]:
                    continue
                elif key == 'restecg' and value not in [0, 1, 2]:
                    continue
                elif key == 'exang' and value not in [0, 1]:
                    continue
                elif key == 'slope' and value not in [1, 2, 3]:
                    continue
                elif key == 'ca' and value not in [0, 1, 2, 3]:
                    continue
                elif key == 'thal' and value not in [3, 6, 7]:
                    continue
                elif key == 'trestbps' and (value < 90 or value > 200):
                    continue
                elif key == 'chol' and (value < 100 or value > 600):
                    continue
                elif key == 'thalach' and (value < 70 or value > 200):
                    continue
                elif key == 'oldpeak' and (value < 0 or value > 6.2):
                    continue
                elif key == 'age' and (value < 20 or value > 100):
                    continue
                
                measurements[key] = value
            except ValueError:
                continue
    
    # Print extracted values for verification
    if measurements:
        print("\nExtracted values from image:")
        for key, value in measurements.items():
            print(f"{key}: {value}")
    
    return measurements

def process_medical_image(image_path: str) -> Dict[str, float]:
    """
    Process a medical report image and extract relevant measurements.

    Args:
        image_path (str): Path to the medical report image

    Returns:
        Dict[str, float]: Dictionary containing extracted medical measurements
    """
    # Load and preprocess image
    image = load_image(image_path)
    
    # Extract text using OCR
    text = extract_text_from_image(image)
    
    # Parse the extracted text
    measurements = parse_medical_report(text)
    
    return measurements

def combine_image_and_tabular_data(
    image_data: Dict[str, float],
    tabular_data: Dict[str, float]
) -> Dict[str, float]:
    """
    Combine measurements extracted from images with tabular data.

    Args:
        image_data (Dict[str, float]): Measurements extracted from images
        tabular_data (Dict[str, float]): Original tabular data

    Returns:
        Dict[str, float]: Combined data dictionary
    """
    combined_data = tabular_data.copy()
    
    # Update with image-extracted measurements where available
    for key, value in image_data.items():
        if key in combined_data:
            # If both sources have the value, use the average
            combined_data[key] = (combined_data[key] + value) / 2
        else:
            combined_data[key] = value
    
    return combined_data 