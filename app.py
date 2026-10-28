import os
import streamlit as st
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral

st.set_page_config(page_title="CareerBot AI", page_icon="🎯")

st.title("🎯 CareerBot AI")
st.caption("Your AI-powered career guidance assistant")

# Load the dataset
@st.cache_data
def load_data():
    data = load_dataset(
        "advy/career-guidance-counsellor-QA",
        split="train"
    )
    frame = data.to_pandas()
    frame.columns = [str(c).strip().lower() for c in frame.columns]

    frame = frame.dropna(subset=["question", "answer"]).copy()
    frame["question"] = frame["question"].astype(str)
    frame["answer"] = frame["answer"].astype(str)

    frame = frame[
        (frame["question"].str.strip() != "") &
        (frame["answer"].str.strip() != "")
    ].reset_index(drop=True)

    return frame

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

st.sidebar.success(f"Loaded {len(df)} career Q&A records")

@st.cache_resource
def build_index(questions, answers):
    documents = [
        f"{q} {a}" for q, a in zip(questions, answers)
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000
    )

    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix

try:
    vectorizer, matrix = build_index(
        tuple(df["question"]),
        tuple(df["answer"])
    )
except Exception as e:
    st.error(f"Search index creation failed: {e}")
    st.stop()

def retrieve(query, top_k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "question": df.iloc[i]["question"],
            "answer": df.iloc[i]["answer"],
            "score": float(scores[i])
        }
        for i in indices if scores[i] > 0
    ]

def generate_answer(query, sources):
    api_key = st.secrets.get(
        "MISTRAL_API_KEY",
        os.environ.get("MISTRAL_API_KEY", "")
    )

    if not api_key:
        raise ValueError(
            "Add MISTRAL_API_KEY to Streamlit Secrets."
        )

    context = "\n\n".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}"
        for s in sources
    )

    if not context:
        context = "No relevant dataset entries were found."

    client = Mistral(api_key=api_key)

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are CareerBot, a practical career counsellor. "
                    "Use the supplied career references when relevant. "
                    "Give clear, actionable advice. Do not invent salaries "
                    "or guarantee employment. If the references are "
                    "insufficient, be transparent."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Career references:\n{context}\n\n"
                    f"My question: {query}\n\n"
                    "Give me a helpful answer with concrete next steps."
                )
            }
        ],
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View dataset references"):
                for source in message["sources"]:
                    st.write(source["question"])
                    st.write(source["answer"])

prompt = st.chat_input("Ask about careers, skills, or internships...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                sources = retrieve(prompt)
                answer = generate_answer(prompt, sources)

            st.markdown(answer)

            if sources:
                with st.expander("View dataset references"):
                    for source in sources:
                        st.markdown(f"**{source['question']}**")
                        st.write(source["answer"])
                        st.caption(
                            f"Similarity: {source['score']:.3f}"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Could not generate an answer: {e}")
