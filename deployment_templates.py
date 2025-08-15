
# ============================================================================
# PRODUCTION DEPLOYMENT CODE TEMPLATES
# ============================================================================

# 1. REAL-TIME STREAMING PROCESSOR (Apache Kafka + Python)
from kafka import KafkaConsumer, KafkaProducer
import json
import logging

class RealTimeIDS:
    def __init__(self, model_path, kafka_config):
        self.model = joblib.load(model_path)
        self.consumer = KafkaConsumer(
            'network_traffic',
            bootstrap_servers=kafka_config['brokers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['brokers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.ids_system = CybersecurityIDS(model, scaler, encoders)
        
    def process_stream(self):
        """Process incoming network traffic stream"""
        for message in self.consumer:
            try:
                record = message.value
                result = self.ids_system.predict_single(record)
                
                if result['final_alert']:
                    alert = {
                        'timestamp': record.get('timestamp'),
                        'session_id': record.get('session_id'),
                        'alert_type': 'INTRUSION_DETECTED',
                        'probability': result['probability'],
                        'confidence': result['confidence'],
                        'details': record
                    }
                    
                    # Send alert to security team
                    self.producer.send('security_alerts', alert)
                    logging.warning(f"ALERT: {alert}")
                    
            except Exception as e:
                logging.error(f"Processing error: {e}")

# 2. REST API FOR BATCH PREDICTIONS (Flask)
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('best_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
ids_system = CybersecurityIDS(model, scaler, encoders)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = ids_system.predict_single(data)
        
        return jsonify({
            'success': True,
            'prediction': int(result['prediction']),
            'probability': float(result['probability']),
            'alert': result['final_alert'],
            'confidence': result['confidence']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        results = []
        
        for record in data['records']:
            result = ids_system.predict_single(record)
            results.append({
                'session_id': record.get('session_id'),
                'prediction': int(result['prediction']),
                'probability': float(result['probability']),
                'alert': result['final_alert']
            })
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# 3. MODEL SERVING WITH MLFLOW
import mlflow
import mlflow.sklearn

class MLflowIDS:
    def __init__(self, model_uri):
        self.model = mlflow.sklearn.load_model(model_uri)
        
    def predict(self, data):
        return self.model.predict(data)

# 4. DOCKER DEPLOYMENT
dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api_server.py"]
"""

# 5. KUBERNETES DEPLOYMENT
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cybersec-ids
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cybersec-ids
  template:
    metadata:
      labels:
        app: cybersec-ids
    spec:
      containers:
      - name: ids-api
        image: cybersec-ids:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.pkl"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: cybersec-ids-service
spec:
  selector:
    app: cybersec-ids
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
"""

print(" Deployment templates created!")
