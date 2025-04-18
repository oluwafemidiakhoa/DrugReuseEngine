import requests
import json

# Base URL for the API
API_BASE_URL = "http://localhost:8000"

# Test the root endpoint
def test_root():
    response = requests.get(f"{API_BASE_URL}/")
    print("Root Endpoint Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

# Test the health endpoint
def test_health():
    response = requests.get(f"{API_BASE_URL}/health")
    print("Health Check Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

# Test authentication
def test_auth():
    # Get a token
    login_data = {
        "username": "admin",
        "password": "password"
    }
    
    response = requests.post(f"{API_BASE_URL}/token", data=login_data)
    print("Authentication Response:")
    
    if response.status_code == 200:
        token_data = response.json()
        print(json.dumps(token_data, indent=2))
        print(f"Status Code: {response.status_code}")
        return token_data["access_token"]
    else:
        print(f"Failed to authenticate: {response.text}")
        print(f"Status Code: {response.status_code}")
        return None

# Test the protected endpoint
def test_protected(token):
    if not token:
        print("Skipping protected endpoint test due to missing token")
        return
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(f"{API_BASE_URL}/protected", headers=headers)
    print("Protected Endpoint Response:")
    print(json.dumps(response.json(), indent=2))
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

# Test drugs endpoint
def test_drugs(token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    response = requests.get(f"{API_BASE_URL}/drugs/", headers=headers)
    print("Drugs Endpoint Response:")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} drugs")
        if data:
            # Print just the first drug
            print("Sample drug:")
            print(json.dumps(data[0], indent=2))
    else:
        print(f"Error: {response.text}")
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

# Test diseases endpoint
def test_diseases(token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    response = requests.get(f"{API_BASE_URL}/diseases/", headers=headers)
    print("Diseases Endpoint Response:")
    if response.status_code == 200:
        data = response.json()
        print(f"Found {len(data)} diseases")
        if data:
            # Print just the first disease
            print("Sample disease:")
            print(json.dumps(data[0], indent=2))
    else:
        print(f"Error: {response.text}")
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

# Test knowledge graph stats
def test_kg_stats(token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    response = requests.get(f"{API_BASE_URL}/knowledge/stats", headers=headers)
    print("Knowledge Graph Stats Response:")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print(f"Status Code: {response.status_code}")
    print("-" * 50)

# Run all tests
if __name__ == "__main__":
    print("==== TESTING API ENDPOINTS ====")
    test_root()
    test_health()
    
    # Get authentication token
    token = test_auth()
    print("-" * 50)
    
    # Test protected endpoint
    test_protected(token)
    
    # Test data endpoints
    test_drugs(token)
    test_diseases(token)
    test_kg_stats(token)
    
    print("==== API TESTING COMPLETE ====")