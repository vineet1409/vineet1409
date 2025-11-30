<!--
Profile README for github.com/vineet1409
-->

<h1 align="center">Hi, I'm Vineet Srivastava 👋</h1>

<p align="center">
  <strong>Senior AI/ML Engineer @ AWS · Agentic AI · Healthcare & Mental-Health AI · Multi‑Cloud MLOps</strong>
</p>

<p align="center">
  <a href="mailto:srivineet93@gmail.com">Email</a> ·
  <a href="https://www.linkedin.com/in/srivastava-vineet">LinkedIn</a> ·
  <a href="https://github.com/vineet1409">GitHub</a> ·
  <a href="https://ai.jmir.org/2025/1/e73448">JMIR AI Paper</a> ·
  <a href="https://doi.org/10.1101/2023.09.25.23296062">MindWatch Preprint</a> ·
  <a href="https://patents.google.com/patent/US20200039784A1/en">Patent</a>
</p>


---

## 🚀 What I’m Doing Now

- **Senior AI/ML Engineer (Delivery Consultant) @ AWS**  
  Building end‑to‑end **agentic AI** solutions for customers using:
  - Amazon **Bedrock**
  - **Strands** agentic framework & **AgentCore** runtime
  - **Knowledge Bases (RAG)**, Gateway/MCP, memory & tools
- Designing **LLM/RAG architectures** with safety, evaluation & observability:
  - Hallucination detection & factuality scoring  
  - Guardrails, prompt orchestration, and retrieval quality evaluation  
  - LLMOps / MLOps pipelines from data to production
- Still deeply involved in **healthcare & mental‑health AI**:
  - Suicide ideation detection, mental‑disorder detection from text
  - Medical information retrieval, diagnosis support, and clinical document understanding

---

## 🎯 Focus Areas

- **Agentic AI on AWS**
  - Multi‑tool agents on Bedrock & Strands, long‑term memory, multi‑step workflows
  - Enterprise‑grade security, monitoring, and cost controls

- **Healthcare & Mental-Health AI**
  - Suicide ideation detection from social media and clinical text  
  - Cancer crowdfunding prediction (linguistic + SDOH features)  
  - Clinical NLP: entity extraction, medical keyword mining, negation handling

- **RAG, LLMOps & MLOps**
  - Vector DBs (FAISS, Chroma, Pinecone), knowledge graphs (Neo4j, Graph RAG)  
  - CI/CD with GitHub Actions, Cloud Run, SageMaker, Azure Web Apps, Databricks  
  - Monitoring, drift, evaluation metrics (ROUGE, BLEU, precision/recall, etc.)

- **IoT & Edge Analytics**
  - BLE connection‑failure prediction for wearables  
  - Battery life / sensor life prediction for smart‑building IoT  
  - Time‑series modeling for localization and device health

---

## 📚 Research & Publications

- **Leveraging LLMs & ML for Cancer Crowdfunding Predictions (JMIR AI 2025)**  
  *Co‑author* – Used GPT‑4o to extract rich linguistic + social determinants of health features from GoFundMe cancer campaigns and combined them with ML (gradient boosting, RF, etc.) for robust success prediction and feature importance analysis.  
  _JMIR AI, 2025;4:e73448_

- **MindWatch: Smart Cloud‑based AI Solution for Suicide Ideation Detection (medRxiv)**  
  *Co‑author & core engineer* – Built an AWS‑hosted system using ALBERT, Bio‑Clinical BERT, Bi‑LSTM, GPT‑3.5, and **LLaMA2** for:
  - Social‑media suicide ideation detection (AUC up to ~0.98 with ALBERT)  
  - Personalized psychoeducation and recommendations via RAG with LLaMA2  
  - Full AWS data‑lake + SageMaker architecture for training & deployment  

- **Talks & Presentations (selected)**  
  - _“MindWatch: Exploring the Potential of Large Language Models for Suicide Ideation Detection”_ – UIC Biostatistics / Psychiatry seminar  
  - Multiple talks on **Generative AI, LLMs, and healthcare** (academic + industry)  
  - Upcoming: podcast appearances (Outlook / Hindustan Times) on practical AI/ML

---

## 🧠 Patent

- **US20200039784A1 – Detecting Elevator Mechanics in Elevator Systems**  
  Co‑inventor on an elevator‑safety system using **UWB tags and anchors** to detect the precise location of mechanics in the hoistway and trigger graded safety actions (alerts, speed limitations, car disable, floor restrictions) across multi‑elevator systems.

---

## 🛠️ Tech Stack (Short Version)

- **Languages:** Python, SQL, R, Embedded C  
- **LLMs & GenAI:** BERT family, GPT‑4/4o/3.5, LLaMA2, Flan‑T5, Sentence Transformers, MedLM, Gemini, HuggingFace ecosystem  
- **RAG / Vector / Graph:** FAISS, Chroma, Pinecone, Neo4j, Graph‑RAG patterns  
- **Cloud:**  
  - **AWS:** S3, Lambda, Glue, Athena, SageMaker, Bedrock, RDS, CloudFormation, CloudWatch, API Gateway  
  - **GCP:** Vertex AI (pipelines, Feature Store, Vector Search), BigQuery, Cloud Run, GKE, GCS  
  - **Azure:** Data Factory, Synapse, Cognitive Search, WebApps, Event Hubs  
- **Data / MLOps:** PySpark, Databricks, MLflow, Docker, Kubernetes, Kafka, GitHub Actions  
- **ML & DL:** classical ML (RF, XGBoost, SVM, etc.), CNNs, LSTMs/RNNs, anomaly detection, SHAP  
- **NLP & CV:** spaCy, NLTK, transformers, OCR (Tesseract), YOLO/OpenCV

(Full detail is in my CV; this is the high‑signal subset.)

---

## 🔍 Selected Projects & Repos

These are some of the more representative projects from my GitHub (including starred repos):

### Mental Health, Healthcare & LLMs

- **[RAG-Mental-Health-Analysis-OpenSourceLLMs](https://github.com/vineet1409/RAG-Mental-Health-Analysis-OpenSourceLLMs)**  
  RAG pipeline for mental‑health text analysis using open‑source LLMs (Python + CSS). Built around vector search + retrieval for explainable mental‑health insights.

- **[generative_ai_mental_health_analysis](https://github.com/vineet1409/generative_ai_mental_health_analysis)**  
  LLM‑powered mental‑disorder detection & recommendations:
  - Uses BERT + OpenAI GPT‑3.5 Turbo and embeddings with FAISS  
  - Streamlit UI, multi‑modal visualizations, and demo videos  
  - Focused on early detection + recommendation flows tied to MindWatch‑style ideas

- **[AI-Med-Assistant](https://github.com/vineet1409/AI-Med-Assistant)**  
  AI medical assistant using open‑source LLMs + RAG:
  - Clinical Q&A using domain documents and embeddings  
  - Web UI + backend built for explainable, source‑linked responses  

- **[healthcare-bigdata-research](https://github.com/vineet1409/healthcare-bigdata-research)**  
  Notebooks and pipelines for healthcare data: EHR‑like datasets, feature engineering, and ML models for diagnostic/symptom analytics.

### LLMOps / MLOps & Production Systems

- **[hands-on-LLMs](https://github.com/vineet1409/hands-on-LLMs)**  
  End‑to‑end LLMOps on **Azure Databricks**:
  - Fine‑tuning LLMs for classification & summarization  
  - Inference, evaluation, and deployment with MLflow  
  - Includes architecture diagrams for RLHF, dbMLops, and Graph RAG pipelines

- **[mlops-project](https://github.com/vineet1409/mlops-project)**  
  Conversational AI for suicide/depression detection:
  - GPT‑3.5‑based app with Streamlit, RAG & CI/CD via GitHub Actions  
  - Deployed to **Azure Web Apps** with automated build/deploy workflow

- **[cloudrun-flask-bigquery](https://github.com/vineet1409/cloudrun-flask-bigquery)**  
  Production‑grade MLOps pipeline demo on **GCP**:
  - Flask app on **Cloud Run**  
  - Loads CSV data from GCS into BigQuery  
  - Uses `uv` for dependency management and GitHub Actions CI/CD

- **[Delta-Live_Tables](https://github.com/vineet1409/Delta-Live_Tables)**  
  Databricks Delta Live Tables experiments:
  - Streaming & batch ETL pipelines  
  - Data quality, expectations, and lineage for analytics & ML

### IoT, Time‑Series & Classical ML

- **[BLE_Connection_failures_Pattern_Prediction](https://github.com/vineet1409/BLE_Connection_failures_Pattern_Prediction)**  
  Predicting Bluetooth connection failures in wearables:
  - Built around real sniffer data and BLE stacks (GAP/GATT/L2CAP)  
  - Pipeline for feature engineering, model training and monitoring

- **[IOT-Sensor-Life_prediction](https://github.com/vineet1409/IOT-Sensor-Life_prediction)**  
  Predicting IoT sensor/battery life in smart buildings:
  - Time‑series + ML pipeline to reduce maintenance costs and optimize deployments

- **[medical_word_embeddings_clincal_trail](https://github.com/vineet1409/medical_word_embeddings_clincal_trail)**  
  Clinical NLP experiments:
  - Medical word embeddings, trial‑related text analysis  
  - Basis for later graph‑RAG and mental‑health modeling work

For more, browse my repos and stars – most of them are either **LLM/RAG/healthcare**, **LLMOps/MLOps**, or **IoT analytics**.

---

## 🤝 How to Reach Me

- 📍 Chicago, IL, USA  
- 📧 Email: **srivineet93@gmail.com**  
- 💼 LinkedIn: [srivastava-vineet](https://www.linkedin.com/in/srivastava-vineet)  
- 🧪 Research:  
  - JMIR AI: Crowdfunding + LLMs  
  - MindWatch medRxiv preprint on suicide ideation detection  
  - Additional work via UIC Biostatistics & Psychiatry publications

If you’re working on **agentic AI, healthcare/mental‑health AI, or serious MLOps problems** and want to collaborate, feel free to reach out.
