
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.analytics import (
    churn_summary,
    churn_by_contract,
    churn_by_tenure,
    monthly_charges,
    churn_rate_by_tenure,
    charges_scatter_data,
    contract_tenure_heatmap,
    customer_churn_share
)

from src.agent import ask_agent


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="ChurnGuard AI",
    layout="wide"
)


# ---------------- DARK THEME ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0e0e0e;
    color: #eeeeee;
}

section[data-testid="stSidebar"] {
    background-color: #151515;
}

div[data-testid="stMetric"] {
    background-color: #191919;
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3, p, label {
    color: #eeeeee !important;
}

.stChatMessage {
    background-color: #191919;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# Matplotlib dark theme
plt.style.use("dark_background")

plt.rcParams["figure.facecolor"] = "#0e0e0e"
plt.rcParams["axes.facecolor"] = "#191919"
plt.rcParams["axes.edgecolor"] = "#555555"
plt.rcParams["text.color"] = "#eeeeee"
plt.rcParams["axes.labelcolor"] = "#eeeeee"
plt.rcParams["xtick.color"] = "#eeeeee"
plt.rcParams["ytick.color"] = "#eeeeee"
plt.rcParams["grid.color"] = "#444444"
plt.rcParams["legend.facecolor"] = "#191919"
plt.rcParams["legend.edgecolor"] = "#555555"


# ---------------- HEADER ----------------

st.title("ChurnGuard AI")
st.caption("Customer churn analytics and intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Chatbot"]
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("Customer Analytics")

    total, churned, rate = churn_summary()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", total)
    c2.metric("Churned Customers", churned)
    c3.metric("Churn Rate", f"{rate:.2f}%")


    # ---------------- BAR CHARTS ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn by Contract")
        st.bar_chart(churn_by_contract())

    with col2:
        st.subheader("Churn Rate by Tenure")
        st.bar_chart(churn_by_tenure())


    # ---------------- MONTHLY CHARGES ----------------

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Average Monthly Charges")
        st.bar_chart(monthly_charges())

    with col4:
        st.subheader("Customer Retention Share")

        share = customer_churn_share()

        fig, ax = plt.subplots(figsize=(5, 4))

        ax.pie(
            share.values,
            labels=share.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#36A2EB", "#FF6384"],
            textprops={"color": "white", "fontsize": 11},
            wedgeprops={
                "width": 0.42,
                "edgecolor": "#0e0e0e",
                "linewidth": 2
            }
        )

        ax.set_title(
            "Retained vs Churned",
            color="white",
            pad=15
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    # ---------------- LINE CHART ----------------

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Churn Rate by Tenure - Line Chart")

        line_data = churn_rate_by_tenure()

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            line_data.index.astype(str),
            line_data.values,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color="#36A2EB"
        )

        ax.set_xlabel("Tenure Group (Months)")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Tenure vs Churn Rate", color="white")
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    # ---------------- SCATTER PLOT ----------------

    with col6:
        st.subheader("Monthly Charges vs Total Charges")

        scatter_data = charges_scatter_data()

        fig, ax = plt.subplots(figsize=(7, 4))

        for label, color, display_name in [
            ("No", "#36A2EB", "Retained"),
            ("Yes", "#FF6384", "Churned")
        ]:

            data = scatter_data[
                scatter_data["Churn"] == label
            ]

            ax.scatter(
                data["MonthlyCharges"],
                data["TotalCharges"],
                alpha=0.65,
                s=22,
                color=color,
                label=display_name
            )

        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Total Charges")
        ax.set_title("Customer Charges Distribution", color="white")
        ax.legend()
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    # ---------------- HEATMAP ----------------

    st.subheader("Contract vs Tenure - Churn Heatmap")

    heatmap_data = contract_tenure_heatmap()

    fig, ax = plt.subplots(figsize=(10, 4))

    chart = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=100
    )

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(
        heatmap_data.columns.astype(str)
    )

    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    ax.set_xlabel("Tenure Group (Months)")
    ax.set_ylabel("Contract Type")
    ax.set_title("Churn Rate by Contract and Tenure", color="white")

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):

            value = heatmap_data.iloc[i, j]

            if pd.notna(value):
                text_color = "white" if value < 65 else "black"

                ax.text(
                    j,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9
                )

    colorbar = fig.colorbar(chart, ax=ax)
    colorbar.set_label("Churn Rate (%)", color="white")
    colorbar.ax.tick_params(colors="white")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    
    # Model Comparison Chart
    st.divider()
    st.header("Model Performance Comparison")

    results = pd.read_csv("models/model_comparison.csv")

    st.bar_chart(
        results.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ]
    )

    st.dataframe(results, use_container_width=True, hide_index=True)



# ==================================================
# CHATBOT
# ==================================================

elif page == "Chatbot":

    st.header("Chatbot")

    st.caption(
        "Ask questions about churn predictions, "
        "customer analytics, and retention strategies."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:
                    answer = ask_agent(query)

                except Exception as e:
                    answer = (
                        "Something went wrong. "
                        "Please check your API key and terminal."
                    )

                    print("Chatbot error:", e)

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
