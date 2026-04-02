from locust import HttpUser, task, between
import random
from PIL import Image
import io
import base64
import os

class ImageClassificationUser(HttpUser):
    wait_time = between(1, 3)  # Simulate user think time
    
    def on_start(self):
        """Load sample images for testing"""
        self.sample_images = []
        # Create dummy images for testing
        for _ in range(10):
            img = Image.new('RGB', (224, 224), color=(random.randint(0,255), 
                                                       random.randint(0,255), 
                                                       random.randint(0,255)))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            self.sample_images.append(img_byte_arr.getvalue())
    
    @task(3)
    def predict_image(self):
        """Send prediction request with random image"""
        image_data = random.choice(self.sample_images)
        files = {
            'file': ('test.jpg', image_data, 'image/jpeg')
        }
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check API health"""
        self.client.get("/health")
    
    @task(1)
    def model_info(self):
        """Get model information"""
        self.client.get("/model-info")