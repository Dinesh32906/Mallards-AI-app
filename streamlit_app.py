import streamlit as st
import snowflake.connector
import pandas as pd
from config import *

# Define the number of chunks to be used in context
num_chunks = 3

def get_snowflake_connection():
    try:
        st.write("Attempting to connect to Snowflake...")
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            role=SNOWFLAKE_ROLE,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA
        )
        st.success("Connected to Snowflake")
        return conn
    except Exception as e:
        st.error(f"Error connecting to Snowflake: {e}")
        return None

def create_prompt(myquestion, rag):
    if rag == 1:
        cmd = f"""
         WITH results AS (
           SELECT RELATIVE_PATH,
             VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
                      SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', '{myquestion.replace("'", "''")}')) AS similarity,
             chunk
           FROM docs_chunks_table
           ORDER BY similarity DESC
           LIMIT {num_chunks}
         )
         SELECT chunk, relative_path FROM results
         """
    
        cur = conn.cursor()
        cur.execute(cmd)
        df_context = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        
        context_length = len(df_context)
        prompt_context = ""
        for i in range(context_length):
            prompt_context += df_context.at[i, 'CHUNK']
        
        prompt_context = prompt_context.replace("'", "")
        relative_path = df_context.at[0, 'RELATIVE_PATH']
        
        prompt = f"""
        'You are an expert assistant extracting information from context provided. 
         Answer the question based on the context. Be concise and do not hallucinate. 
         If you don’t have the information, just say so.
        Context: {prompt_context}
        Question: {myquestion}
        Answer:'
        """
        
        cmd2 = f"SELECT GET_PRESIGNED_URL(@docs, '{relative_path}', 360) AS URL_LINK FROM directory(@docs)"
        cur.execute(cmd2)
        df_url_link = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
        url_link = df_url_link.at[0, 'URL_LINK']
    
    else:
        prompt = f"""
        'Question: {myquestion}
        Answer:'
        """
        url_link = "None"
        relative_path = "None"
        
    return prompt, url_link, relative_path

def complete(myquestion, model_name, rag=1):
    prompt, url_link, relative_path = create_prompt(myquestion, rag)
    cmd = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE('{model_name}', {prompt}) AS response
    """
    
    cur = conn.cursor()
    cur.execute(cmd)
    df_response = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
    return df_response, url_link, relative_path

def display_response(question, model, rag=0):
    response, url_link, relative_path = complete(question, model, rag)
    res_text = response.at[0, 'RESPONSE']
    st.markdown(res_text)
    if rag == 1:
        display_url = f"Link to [{relative_path}]({url_link}) that may be useful"
        st.markdown(display_url)

# Main code
st.title("Snowflake Cortex Document Assistant")
st.write("""You can ask questions and decide if you want to use your documents for context or allow the model to create their own response.""")
st.write("This is the list of documents you already have:")

conn = get_snowflake_connection()
if conn:
    st.write("Connection established.")
    cur = conn.cursor()
    cur.execute("ls @docs")
    docs_available = cur.fetchall()
    list_docs = [doc[0] for doc in docs_available]
    st.dataframe(list_docs)

    # Here you can choose what LLM to use. Please note that they will have different cost & performance
    model = st.sidebar.selectbox('Select your model:', (
                                    'mixtral-8x7b',
                                    'snowflake-arctic',
                                    'mistral-large',
                                    'llama3-8b',
                                    'llama3-70b',
                                    'reka-flash',
                                     'mistral-7b',
                                     'llama2-70b-chat',
                                     'gemma-7b'))

    question = st.text_input("Enter question", placeholder="Ask anything about event...", label_visibility="collapsed")

    rag = st.sidebar.checkbox('Use your own documents as context?')

    if st.button('Submit'):
        if question:
            display_response(question, model, rag)
        else:
            st.warning("Please enter a question.")
else:
    st.error("Failed to connect to Snowflake.")
