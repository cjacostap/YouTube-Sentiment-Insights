#!/usr/bin/env python3
"""
Script de prueba para verificar que la API Flask funciona correctamente.
Ejecuta este script mientras la API está corriendo en otra terminal.
"""

import requests
import json

# URL base de la API
BASE_URL = "http://127.0.0.1:5001"

def test_home():
    """Prueba el endpoint raíz."""
    print("Testing / endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {response.text}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_predict():
    """Prueba el endpoint de predicción."""
    print("\nTesting /predict endpoint...")
    try:
        test_comments = [
            "This video is amazing! I love it!",
            "This is terrible, worst video ever",
            "It's okay, nothing special"
        ]
        
        payload = {"comments": test_comments}
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response:")
        results = response.json()
        for item in results:
            print(f"  - Comment: '{item['comment'][:50]}...'")
            print(f"    Sentiment: {item['sentiment']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Flask API")
    print("=" * 60)
    print(f"Make sure the API is running on {BASE_URL}")
    print("=" * 60)
    
    results = []
    results.append(("Home endpoint", test_home()))
    results.append(("Predict endpoint", test_predict()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")


