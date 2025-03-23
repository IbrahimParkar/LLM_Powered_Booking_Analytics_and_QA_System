import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from RAG import (
    rag_pipeline,
    get_analytics,
)

app = Flask(__name__)

# Load FAISS index and embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index("booking_faiss.index")
    print("FAISS index and embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS or model: {e}")


# SQLite helper function
def get_db_connection():
    try:
        conn = sqlite3.connect("analytics.db")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


# Analytics endpoint
@app.route("/analytics", methods=["POST"])
def analytics():
    try:
        analytics_data = get_analytics()
        return jsonify(analytics_data)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch analytics: {str(e)}"}), 500


# Natural Language Query (RAG) endpoint
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Use the RAG pipeline to handle the query
        response = rag_pipeline(query)

        # Remove unwanted 'Response:\n' from the beginning of the response
        formatted_response = response.strip().replace("Response:\n", "").replace("\\n", "\n").replace("\\t", "\t")

        return jsonify({"response": formatted_response})
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500


# System Health Check endpoint
@app.route("/health", methods=["GET"])
def health():
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({"status": "healthy", "message": "All systems operational"})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
