# Tactical Football Event Classifier 


## Project Overview
 This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to automate the tagging and classification of tactical football events from static imagery. 

By classifying key match events, this tool assists football scouts and tactical analysts in evaluating a team's set-piece organization and player discipline without manually scrubbing through hours of match footage.

The project features a fine-tuned Deep Learning model (MobileNetV2) served via a **FastAPI** backend and consumed by an interactive **Streamlit** frontend dashboard, all containerized using **Docker** for seamless deployment.

---

## Tactical Classes
The model has been explicitly trained on a curated subset of 2,000 images to identify four distinct tactical events:
*   **Corner:** Evaluates attacking/defending set-piece structures.
*   **Free Kick:** Analyzes defensive walls and offensive setups.
*   **Goal Attempt:** Captures chaotic penalty-box scenarios and goalkeeper reactions.
*   **Yellow Card:** Tracks player discipline and match aggression.

*(Note: The generalized "Goal" class was deliberately excluded from this pipeline to prevent visual overlap and confusion with "Goal Attempts", ensuring high-fidelity tactical predictions).*

---

## Model Architecture & Performance
The core computer vision engine utilizes a **MobileNetV2** architecture, chosen for its lightweight footprint and high efficiency in production inference environments.

### Test Set Evaluation & Tactical Insights
The model achieved an outstanding **90.75% overall accuracy** on the unseen test holdout. Analyzing the classification report reveals fascinating insights into how the CNN interprets tactical football imagery:

1.  **High-Contrast Feature Extraction (Yellow Cards):** The model achieved near-perfect metrics (98% F1-Score) on Yellow Cards. This indicates the network successfully isolated the distinct color thresholding and referee posture, cleanly separating it from standard gameplay.
2.  **The Free Kick / Goal Attempt Overlap:** The model showed a slight hesitation between Free Kicks (74% Recall) and Goal Attempts (80% Precision). This is a known thresholding challenge in sports computer vision; both events feature densely packed players in the penalty box, defensive walls, and goalkeepers in high-alert postures. The model conservatively leans towards predicting "Goal Attempt" when presented with dense penalty-box formations.

### Addressing the Generalization Gap
During training, the training accuracy climbed to ~99.6% while the validation accuracy plateaued at ~93%. Rather than aggressively applying dropout layers (which would simply drag training accuracy down without improving real-world performance), the validation loss was monitored closely. Because the validation loss steadily decreased across all epochs without spiking, the early stopping callback allowed the model to converge safely, confirming that the 93% validation accuracy is a robust, production-ready metric.

---

## MLOps System Architecture

This project is deployed using a microservices architecture, heavily optimized for inference speed and low resource consumption.

*   **Model Training Environment:** TensorFlow (Full / GPU enabled) via Jupyter Notebooks.
*   **Production Inference Environment:** The deployment utilizes `python:3.10-slim` base images and `tensorflow-cpu` to reduce container bloat by over 80%, ensuring blazing-fast build times and low cloud hosting costs.
*   **Backend:** FastAPI routes image payloads to the frozen `.h5` model and returns JSON predictions.
*   **Frontend:** Streamlit provides a user-friendly UI for scouts to upload images and view real-time model confidence visualizations.

---

##  How to Run the Application

This application is fully containerized and designed to run effortlessly in **GitHub Codespaces** or any local Docker Desktop environment.

### Step 1: Clone the Repository
git clone https://github.com/YinkaAjao/Tactical-Video-Analyzer.git 
cd Tactical-Video-Analyzer

### Step 2: Build and Spin Up the Containers
Ensure Docker is running, then execute:

docker-compose up --build -d
### Step 3: Access the Platform
Once the containers are successfully built and running, open your web browser and navigate to:

Streamlit UI (Frontend): http://localhost:8501

FastAPI Docs (Backend): http://localhost:8000/docs

To Stop the Application:

docker-compose down
