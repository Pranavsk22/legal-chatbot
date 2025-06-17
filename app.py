import streamlit as st
from generator.rag_pipeline import retrieve_context, generate_answer

st.set_page_config(page_title="Indian Legal Chatbot", layout="centered")
st.title("⚖️ AI Legal Assistant (India)")

query = st.text_input("Enter your legal question:")

if query:
    with st.spinner("Thinking like a lawyer..."):
        context_chunks = retrieve_context(query)
        answer = generate_answer(query, context_chunks)

        st.subheader("📚 Retrieved Context")
        for i, chunk in enumerate(context_chunks, 1):
            st.markdown(f"**{i}.** {chunk.strip()}")

        st.subheader("💬 AI Answer")
        st.markdown(answer)
        st.subheader("This is not legal advice. Please consult a licensed lawyer for actual legal matters.")
