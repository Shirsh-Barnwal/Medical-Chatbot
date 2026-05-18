from flask import Flask, render_template, request

from src.helper import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

import google.generativeai as genai
import os


# Flask app
app = Flask(__name__)


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Configure APIs
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)


# Gemini model
model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)


# Embedding model
embeddings = download_embeddings()


# Pinecone index
index_name = "medical-chatbot"


# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Home route
@app.route("/")
def index():
    return render_template("chat.html")


# Chat route
@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("User Query:", msg)

    # Retrieve documents
    docs = retriever.invoke(msg)

    # Create context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Final prompt
    final_prompt = system_prompt.format(
        context=context,
        question=msg
    )

    # Generate response
    response = model.generate_content(final_prompt)

    answer = response.text

    print("Response:", answer)

    return str(answer)


# Run app
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )