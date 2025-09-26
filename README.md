# MLally

MLally is a no-code / low-code Machine Learning platform designed to let users train models from CSV datasets, do preprocessing, and deploy models with an API endpoint.  
It aims to simplify the process of going from raw data → model → prediction without writing a lot of boilerplate.

---

## 🚀 Features

- Upload a CSV and do basic preprocessing (e.g. cleaning, handling missing values — via `pre.py` etc.)  
- Train machine learning models automatically with minimal configuration  
- Utility scripts / modules for model training, evaluation, and saving/loading models  
- A server architecture: separate components for client-server, model-server, buffer server etc.  
- API / web interface to get predictions once model is deployed  

---

## 🛠️ Tech Stack

- **Python 3.x**  
- **TensorFlow / PyTorch / scikit-learn** (depending on what model_utils uses)  
- **FastAPI / Flask** (or similar) for serving the model as an API server  
- **Client-Server architecture** (main_client_server, model_server, buffer_server)  
- **Utilities for preprocessing** (`pre.py`)  
- Data handling with **Pandas**, numerical computation with **NumPy**  

---



