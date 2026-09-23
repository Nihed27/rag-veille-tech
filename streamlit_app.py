import os
import requests
import streamlit as st

# URL de l'API FastAPI. En local (hors Docker) : http://127.0.0.1:8000
# Dans docker-compose : http://api:8000 (nom du service, pas une IP)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Veille Tech", page_icon="📚")

st.title("📚 RAG Veille Tech")
st.caption("Pose une question sur ton cours (Analyse de Données).")

question = st.text_input("Ta question :", placeholder="Ex: Qu'est-ce que l'ACP ?")

if st.button("Envoyer") and question.strip():
    with st.spinner("Recherche en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Réponse")
            st.write(data.get("answer", "Pas de réponse."))

            verdict = data.get("verdict")
            if verdict:
                if verdict == "ok":
                    st.success(f"Verdict de l'agent critique : {verdict}")
                else:
                    st.warning(f"Verdict de l'agent critique : {verdict}")

            sources = data.get("sources", [])
            if sources:
                st.subheader("Sources")
                for s in sources:
                    st.markdown(f"- **{s.get('source')}** — page {s.get('page')}")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")