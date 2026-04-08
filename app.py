import streamlit as st
import pandas as pd
import json
from sdv.single_table import CTGANSynthesizer
import google.generativeai as genai

# -------------------------------
# 🔑 SET GEMINI API KEY
# -------------------------------
genai.configure(api_key="AIzaSyDBSR2EDAgd19LcHaDRfAckTi1AXbtJffE")
llm_model = genai.GenerativeModel("models/gemini-2.5-flash")

# -------------------------------
# 📦 LOAD CTGAN MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return CTGANSynthesizer.load("ctgan_healthcare.pkl")

ctgan_model = load_model()

# -------------------------------
# 🤖 LLM MULTI-COLUMN FUNCTION
# -------------------------------
def generate_multiple_columns(df, user_prompt, selected_columns):

    # Keep only selected columns
    df_filtered = df[selected_columns]

    prompt = f"""
    You are a healthcare assistant.

    Dataset:
    {df_filtered.to_dict(orient='records')}

    Task:
    {user_prompt}

    

    STRICT RULES:
    - Return ONLY JSON
    - Output must be a list of objects
    - No extra text

    Example:
    [
      {{"risk": "High", "advice": "Exercise more"}},
      {{"risk": "Low", "advice": "Maintain lifestyle"}}
    ]
    """

    response = llm_model.generate_content(prompt)

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    outputs = json.loads(text)

    new_cols_df = pd.DataFrame(outputs)

    # 🔥 Only selected + generated columns
    final_df = pd.concat([df_filtered.reset_index(drop=True), new_cols_df], axis=1)

    return final_df

# -------------------------------
# 🎨 UI
# -------------------------------
st.title("MEDGENAI")

# Get columns from sample
sample_df = ctgan_model.sample(num_rows=5)
columns = list(sample_df.columns)

# Inputs
num_rows = st.number_input("Number of rows", 5, 100, 10)

selected_columns = st.multiselect(
    "Select columns to include",
    options=columns
)

user_prompt = st.text_area(
    "What new columns do you want?",
    placeholder="e.g., Give risk_level, explanation, and lifestyle_advice"
)

# Button
if st.button("Generate Data"):

    st.write("⏳ Generating synthetic data...")

    synthetic_df = ctgan_model.sample(num_rows=num_rows)

    # 🔥 Logic for output
    if selected_columns:
        if user_prompt.strip() != "":
            try:
                final_df = generate_multiple_columns(
                    synthetic_df,
                    user_prompt,
                    selected_columns
                )
            except:
                st.error("LLM failed. Showing selected columns only.")
                final_df = synthetic_df[selected_columns]
        else:
            final_df = synthetic_df[selected_columns]
    else:
        final_df = synthetic_df

    st.success("✅ Done!")

    st.dataframe(final_df)

    # Download button
    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "synthetic_data.csv", "text/csv")