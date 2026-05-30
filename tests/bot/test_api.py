import unittest
import os
import io
from fastapi.testclient import TestClient
from phoenix_api import app

class PhoenixApiTest(unittest.TestCase):
    """
    Test suite for the Phoenix AI FastAPI server.
    Tests /health and /chat endpoints using TestClient.
    """
    
    @classmethod
    def setUpClass(cls):
        # We use a context manager to ensure startup/shutdown events (like bot init) run
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_health_endpoint(self):
        """Test the /health heartbeat endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "alive")
        self.assertEqual(response.json()["service"], "Phoenix AI")

    def test_chat_text_only(self):
        """Test the /chat endpoint with text only."""
        data = {
            "prompt": "Hello Phoenix, how are you today?",
            "session_id": "test_session_123"
        }
        # /chat uses Form data
        response = self.client.post("/chat", data=data)
        
        if response.status_code == 500:
            print(f"\n[!] 500 Error Detail: {response.json().get('detail')}")
        elif response.status_code == 503:
            print(f"\n[!] 503 Initialization Error: {response.json().get('detail')}")
            
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "success")
        self.assertIn("reply", json_data)
        self.assertEqual(json_data["session_id"], "test_session_123")

    def test_chat_with_image(self):
        """Test the /chat endpoint with an image upload."""
        # Create a dummy image in memory
        image_content = b"fake image content"
        image_file = io.BytesIO(image_content)
        
        data = {
            "prompt": "What is in this image?",
            "session_id": "vision_session"
        }
        files = {
            "image": ("test.jpg", image_file, "image/jpeg")
        }
        
        response = self.client.post("/chat", data=data, files=files)
        
        if response.status_code == 500:
            print(f"\n[!] 500 Error Detail (Image): {response.json().get('detail')}")
        elif response.status_code == 503:
            print(f"\n[!] 503 Initialization Error (Image): {response.json().get('detail')}")

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "success")
        self.assertIn("reply", json_data)

    def test_chat_missing_prompt(self):
        """Test error handling when prompt is missing."""
        data = {
            "session_id": "error_session"
        }
        response = self.client.post("/chat", data=data)
        
        # FastAPI returns 422 Unprocessable Entity for missing required fields
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()

