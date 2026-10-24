
import os
import json
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from src.data_processing import load_data
from src.analytics import churn_summary
from src.rag_search import search_knowledge

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# TOOL 1: CUSTOMER ANALYTICS

def analyze_customers(group_by="Contract"):

    df = load_data()

    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    rate = churned / total * 100 if total else 0

    # Group tenure into simple categories
    if group_by == "tenure":
        df["tenure"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=[
                "0-12 Months",
                "13-24 Months",
                "25-48 Months",
                "49-72 Months"
            ]
        )

    # Count total and churned customers in each group
    grouped = df.groupby(group_by, dropna=False)

    result = []

    for name, group in grouped:

        total_in_group = len(group)
        churned_in_group = (group["Churn"] == "Yes").sum()

        churn_rate = (
            churned_in_group / total_in_group * 100
            if total_in_group > 0 else 0
        )

        result.append({
            "category": str(name),
            "total_customers": int(total_in_group),
            "churned_customers": int(churned_in_group),
            "churn_rate": round(churn_rate, 2)
        })

    result = sorted(
        result,
        key=lambda x: x["churned_customers"],
        reverse=True
    )

    return {
        "total_customers": int(total),
        "churned_customers": int(churned),
        "churn_rate": round(rate, 2),
        "group_by": group_by,
        "groups": result
    }


# TOOL 2: RAG KNOWLEDGE SEARCH

def search_retention_knowledge(query):

    results = search_knowledge(query, k=5)

    if not results:
        return "Relevant information was not found in the knowledge base."

    context = ""

    for result in results:

        if isinstance(result, dict):
            source = result.get("source", "Unknown source")
            content = result.get("text", str(result))
            context += f"\nSource: {source}\n{content}\n"

        else:
            context += str(result) + "\n"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": """
Answer using only the provided knowledge base.

Use simple English and clear bullet points.
Explain the main factors influencing telecom customer churn.
Ignore unrelated research questions and irrelevant text.
Do not invent facts.
Mention the PDF source names when available.
If the information is insufficient, say so.
"""
            },
            {
                "role": "user",
                "content": (
                    "Question: " + query +
                    "\n\nKnowledge Base:\n" + context
                )
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# TOOL 3: ML CHURN PREDICTION

def predict_churn(customer_data):

    base = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model = joblib.load(
        os.path.join(base, "models", "churn_model.pkl")
    )

    scaler = joblib.load(
        os.path.join(base, "models", "scaler.pkl")
    )

    columns = joblib.load(
        os.path.join(base, "models", "columns.pkl")
    )

    train_data = load_data()

    train_data = train_data.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    row = pd.DataFrame([customer_data])

    row["TotalCharges"] = pd.to_numeric(
        row["TotalCharges"], errors="coerce"
    )

    for col in row.columns:

        if col in train_data.columns and row[col].isnull().any():

            if pd.api.types.is_numeric_dtype(train_data[col]):
                row[col] = row[col].fillna(
                    train_data[col].median()
                )

            else:
                row[col] = row[col].fillna(
                    train_data[col].mode()[0]
                )

    combined = pd.concat(
        [train_data, row],
        ignore_index=True
    )

    combined = pd.get_dummies(
        combined,
        drop_first=True,
        dtype=int
    )

    x = combined.iloc[[-1]].copy()
    x = x.reindex(columns=list(columns), fill_value=0)
    x = x.fillna(0)

    x = scaler.transform(x)

    prediction = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1]) * 100

    return {
        "prediction": (
            "Likely to Churn" if prediction == 1
            else "Likely to Stay"
        ),
        "churn_probability": round(probability, 2)
    }


# AVAILABLE AGENT TOOLS

tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_customers",
            "description": (
                "Analyze actual customer data. Use this for questions "
                "about churn counts or churn rates by gender, contract, "
                "payment method, internet service, tenure, and other "
                "supported categories. Select the matching group_by."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": [
                            "gender",
                            "Contract",
                            "tenure",
                            "InternetService",
                            "PaymentMethod",
                            "SeniorCitizen",
                            "Partner",
                            "Dependents",
                            "TechSupport",
                            "OnlineSecurity",
                            "PaperlessBilling",
                            "PhoneService",
                            "MultipleLines",
                            "OnlineBackup",
                            "DeviceProtection",
                            "StreamingTV",
                            "StreamingMovies"
                        ],
                        "description": "Dataset column to analyze"
                    }
                },
                "required": ["group_by"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_retention_knowledge",
            "description": (
                "Search the PDF knowledge base for telecom churn reasons "
                "and customer retention strategies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": (
                "Predict churn for an individual customer. Use only when "
                "the user provides all required customer details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_data": {
                        "type": "object",
                        "properties": {
                            "gender": {
                                "type": "string",
                                "enum": ["Female", "Male"]
                            },
                            "SeniorCitizen": {
                                "type": "integer",
                                "enum": [0, 1]
                            },
                            "Partner": {
                                "type": "string",
                                "enum": ["Yes", "No"]
                            },
                            "Dependents": {
                                "type": "string",
                                "enum": ["Yes", "No"]
                            },
                            "tenure": {"type": "integer"},
                            "PhoneService": {
                                "type": "string",
                                "enum": ["Yes", "No"]
                            },
                            "MultipleLines": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No phone service"
                                ]
                            },
                            "InternetService": {
                                "type": "string",
                                "enum": ["DSL", "Fiber optic", "No"]
                            },
                            "OnlineSecurity": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No internet service"
                                ]
                            },
                            "OnlineBackup": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No internet service"
                                ]
                            },
                            "DeviceProtection": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No internet service"
                                ]
                            },
                            "TechSupport": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No internet service"
                                ]
                            },
                            "StreamingTV": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No internet service"
                                ]
                            },
                            "StreamingMovies": {
                                "type": "string",
                                "enum": [
                                    "Yes", "No", "No internet service"
                                ]
                            },
                            "Contract": {
                                "type": "string",
                                "enum": [
                                    "Month-to-month",
                                    "One year",
                                    "Two year"
                                ]
                            },
                            "PaperlessBilling": {
                                "type": "string",
                                "enum": ["Yes", "No"]
                            },
                            "PaymentMethod": {
                                "type": "string",
                                "enum": [
                                    "Electronic check",
                                    "Mailed check",
                                    "Bank transfer (automatic)",
                                    "Credit card (automatic)"
                                ]
                            },
                            "MonthlyCharges": {"type": "number"},
                            "TotalCharges": {"type": "number"}
                        },
                        "required": [
                            "gender",
                            "SeniorCitizen",
                            "Partner",
                            "Dependents",
                            "tenure",
                            "PhoneService",
                            "MultipleLines",
                            "InternetService",
                            "OnlineSecurity",
                            "OnlineBackup",
                            "DeviceProtection",
                            "TechSupport",
                            "StreamingTV",
                            "StreamingMovies",
                            "Contract",
                            "PaperlessBilling",
                            "PaymentMethod",
                            "MonthlyCharges",
                            "TotalCharges"
                        ]
                    }
                },
                "required": ["customer_data"]
            }
        }
    }
]


# MAIN AGENT

def ask_agent(query):

    query_lower = query.lower().strip()

    # Allow simple greetings
    if query_lower in [
        "hi", "hello", "hey", "hello there",
        "good morning", "good afternoon", "good evening"
    ]:
        return (
            "Hello! I am ChurnGuard AI. I can help with customer "
            "analytics, churn predictions, retention knowledge, and reports."
        )

    # Basic unrelated question filter
    allowed_topics = [
        "churn", "customer", "retention", "contract",
        "tenure", "charges", "prediction", "predict",
        "analytics", "dataset", "telecom", "telco",
        "cancellation", "service", "payment", "knowledge base",
        "rag", "model", "risk", "report", "loyalty",
        "subscriber", "subscription", "leaving", "leave",
        "cancel", "stay", "lost customers", "customer loss",
        "identify", "detect", "reduce", "prevent",
        "why customers", "customer behavior", "customer behaviour",
        "churn rate", "churn analysis", "churn prediction",
        "churn model", "retention strategy", "retention strategies",
        "churnguard"
    ]

    if not any(topic in query_lower for topic in allowed_topics):
        return (
            "I can only answer questions related to ChurnGuard AI, "
            "customer churn analytics, churn prediction, and customer "
            "retention strategies. Please ask a relevant question."
        )

    messages = [
        {
            "role": "system",
            "content": """
You are ChurnGuard AI, a telecom customer churn assistant.

Rules:
- Answer questions related to telecom customer churn,
  analytics, prediction, retention, and this project's tools.
- For dataset statistics, call analyze_customers.
- For gender-wise churn, use group_by="gender".
- For contract-wise churn, use group_by="Contract".
- For payment method questions, use group_by="PaymentMethod".
- For internet service questions, use group_by="InternetService".
- For tenure questions, use group_by="tenure".
- For other supported categories, select the matching column.
- If asked for the highest number of churns, compare churned_customers.
- If asked for the highest churn rate or percentage, compare churn_rate.
- Answer with the actual results returned by the tool.
- Never invent counts, percentages, or prediction results.
- Use search_retention_knowledge for PDF research questions.
- Use predict_churn only with a complete customer profile.
- If customer details are missing, ask the user for them.
- Keep answers simple and clear.
"""
        },
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    if not message.tool_calls:
        return message.content or "I could not generate an answer."

    answers = []

    for tool_call in message.tool_calls:

        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        try:

            if name == "analyze_customers":

                result = analyze_customers(
                    args.get("group_by", "Contract")
                )

                answer = (
                    f"Total customers: {result['total_customers']}\n"
                    f"Churned customers: {result['churned_customers']}\n"
                    f"Overall churn rate: {result['churn_rate']}%\n\n"
                    f"Churn by {result['group_by']}:\n"
                )

                for group in result["groups"]:
                    answer += (
                        f"- {group['category']}: "
                        f"{group['churned_customers']} churned out of "
                        f"{group['total_customers']} customers "
                        f"({group['churn_rate']}% churn rate)\n"
                    )

                if result["groups"]:
                    top = result["groups"][0]

                    answer += (
                        f"\nHighest churn count: {top['category']} "
                        f"with {top['churned_customers']} customers."
                    )

            elif name == "search_retention_knowledge":

                answer = search_retention_knowledge(args["query"])

            elif name == "predict_churn":

                result = predict_churn(args["customer_data"])

                answer = (
                    f"Prediction: {result['prediction']}\n"
                    f"Churn probability: "
                    f"{result['churn_probability']}%"
                )

            else:
                answer = "Unknown tool requested."

        except Exception as e:
            answer = f"Tool error in {name}: {str(e)}"

        answers.append(answer)

    return "\n\n".join(answers)
