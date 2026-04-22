import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="FinSight AI", page_icon="📈")

st.title("📈 FinSight AI: Financial Query Engine")
st.markdown("Query investment strategies and financial data powered by RAG.")

# 2. Get Backend URL from Secrets (We will set this in Step 4)
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except:
    st.error("Backend URL not found in Streamlit Secrets.")
    st.stop()

# 3. User Input
user_query = st.text_input("Enter your financial question:")

if st.button("Analyze"):
    if user_query:
        with st.spinner("FinSight is analyzing data..."):
            try:
                # This sends the query to your Render API
                response = requests.post(
                    f"{BACKEND_URL}/query", 
                    json={"query": user_query},
                    timeout=90  # Longer timeout to allow Render to "wake up"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Analysis Complete")
                    st.write(data.get("answer", "No answer found in response."))
                else:
                    st.error(f"Backend Error: {response.status_code}")
                    st.info("Note: If the backend was asleep, please wait 60 seconds and try again.")
                    
            except requests.exceptions.RequestException as e:
                st.error("Could not connect to the backend.")
                st.info("Your Render backend might be 'spinning up'. Please wait a minute and refresh.")
    else:
        st.warning("Please enter a query first.")