# ChurnGuard AI

### AI-Powered Telecom Customer Churn Prediction & Retention Assistant

ChurnGuard AI is a Machine Learning and Generative AI project built to help understand telecom customer churn identify customers who may leave and explore ways to improve customer retention.

The project brings together a Streamlit dashboard, a Machine Learning model, a document-based Retrieval-Augmented Generation (RAG) system and an AI agent. These components help users explore customer data understand churn patterns and get insights through a chatbot.

**Live Webpage Link for project:**

https://ishankhan07-churnguard-ai-app-nnayjt.streamlit.app/

## Dashboard & Customer Analytics

The dashboard gives users a view of customer churn through KPIs, charts, customer analytics and machine learning model evaluation. It helps make customer data easier to understand and explore.

### Key Features

- View customer and churn summaries through KPIs

- Explore churn patterns across different customer segments

- Use bar, donut, line and scatter charts

- Ask questions in natural language to explore analytics

- Compare different Machine Learning models

- View model evaluation visualizations

### Dashboard Screenshots

**1. KPI Dashboard**

![KPI Dashboard](screenshots/Kpi_dashboard.png)

**2. Customer Analytics**

![Customer Analytics](screenshots/analytics_que.png)

**3. Customer Churn Insights**

![Customer Churn Insights](screenshots/insight_que.png)

**4.. Donut Charts**

![Bar and Donut Charts](screenshots/bar_donut.png)

**5.. Scatter Charts**

![Line and Scatter Charts](screenshots/line_scatter.png)

**6. Machine Learning Model Comparison**

![Model Comparison](screenshots/model_compare.png)

## AI Chatbot

ChurnGuard AI also includes a chatbot where users can ask questions about customer churn, analytics, machine learning predictions and customer retention.

The chatbot brings together three capabilities:

### 1. Machine Learning. Churn Prediction

The trained classification model estimates whether a telecom customer is likely to churn using details such as tenure, contract type, monthly charges and subscribed services.

**How it helps:**

- Identifies customers who may be at a risk of churning

- Helps with customer retention planning

- Allows businesses to focus on customers who may need additional attention

![Machine Learning Churn Prediction](screenshots/ml_ques.png)

### 2. Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) helps the chatbot find information from a knowledge base before generating a response.

For this project the knowledge base contains documents related to telecom churn and customer retention. This allows the chatbot to use information retrieved from these documents when answering questions.

**RAG workflow:**

1.. Process PDF documents.

2. Split the document text into chunks.

3. Generate embeddings using a sentence-transformer model.

4. Store and search document embeddings using FAISS.

5. Retrieve text based on the users question.

6. Use the retrieved context to generate a response with a Large Language Model (LLM).

This approach helps connect the chatbots answers to information found in the projects knowledge base of relying only on the LLMs general knowledge.

**RAG Example 1**

![RAG Example 1](screenshots/rag_ques1.png)

**RAG Example 2**

![RAG Example 2](screenshots/rag_que2.png)

### 3. AI Agent

The AI agent connects project capabilities through tools. It can use analytics, machine learning predictions and retrieved knowledge to help answer business-related questions.

The agent can use tools to:

- `analyze_customers()`. Analyze customer churn rates and customer segments.

- `predict_churn()`. Use the trained Machine Learning model to predict churn risk.

- `search_retention_knowledge()`. Search the RAG knowledge base for information.

By bringing these tools the agent can provide more useful responses to questions about customer churn and retention.

## Machine Learning Workflow

The Machine Learning component follows a classification workflow from preparing the customer data to using the trained model for predictions.

1. Load the telecom customer churn dataset.

2. Separate the customer. Target variable.

3.. Process the data for model training.

4.. Compare classification models.

5. Evaluate the performance of the models.

6. Save the selected model and the required preprocessing objects.

7. Use the trained model to predict churn risk within the application.

Model comparison and evaluation visualizations help users review how the models perform.

## Tech Stack

| Category | Technologies |

|---|---|

| Programming Language | Python |

| Data Analysis | Pandas, NumPy |

| Visualization | Matplotlib, Seaborn |

| Machine Learning | Scikit-learn |

| Imbalanced Data Handling | Imbalanced-learn |

| Generative AI | Groq API |

| Embeddings | Sentence Transformers |

| Vector Search | FAISS |

| RAG Framework | LangChain |

PDF Processing | PyPDF |

| Frontend & Dashboard | Streamlit |

| Model Storage | Joblib |

## Project Structure

```text

ChurnGuard-AI/

│

├── app.py

├── requirements.txt

├── README.md

├──.gitignore

│

├── data/

│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv

│

├── knowledge_base/

│   ├── 858027226-DSS-2-DRAFT.pdf

│   ├── A comprehensive survey on customer churn analysis studies.pdf

│   ├── Determinants of churn in telecommunication.pdf

│   ├── retention and churn prediction.pdf

│   ├── chunks.pkl

│   └── faiss_index.bin

│

├── models/

│   ├── churn_model.pkl

│   ├── columns.pkl

│   ├── scaler.pkl

│   └── model_comparison.csv

│

├── screenshots/

│   ├── Kpi_dashboard.png

│   ├── analytics_que.png

│   ├── insight_que.png

│   ├── bar_donut.png

│   ├── line_scatter.png

│   ├── model_compare.png

│   ├── ml_ques.png

│   ├── rag_ques1.png

│   └── rag_que2.png

│

└── src/

├── agent.py

├── analytics.py

├── data_processing.py

├── ml_model.py

├── rag.py

└── rag_search.py

```

## How to Run Locally

### 1. Clone the Repository

```bash

git clone https://github.com/Ishankhan07/ChurnGuard-Ai.git

cd ChurnGuard-Ai

```

### 2. Install Dependencies

```bash

pip install -r requirements.txt

```

### 3. Configure the API Key

Create a `.env` file in the project root and add your Groq API key:

```text

GROQ_API_KEY=your_groq_api_key

```

Keep this file private. Never upload an API key to GitHub.

### 4. Run the Application

```bash

streamlit run app.py

```

Once the application starts, open the URL shown in your terminal to access the dashboard and chatbot.

## Project Objective

ChurnGuard AI brings Machine Learning, customer analytics, RAG and AI-agent tools together to address a problem, in the telecom industry: customer churn.

The aim is to make customer data and domain knowledge more useful by turning them into insights that can help businesses understand churn patterns and make customer-retention decisions.

---

**Built with Python Machine Learning, RAG and Streamlit.**

Mohammad Ishan Khan
