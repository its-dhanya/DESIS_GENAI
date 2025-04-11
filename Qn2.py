import os
import logging
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__, template_folder="templates")
app.debug = True

logging.basicConfig(level=logging.DEBUG)
def load_documents(directory):
    documents = []
    if not os.path.exists(directory):
        app.logger.error(f"Directory {directory} does not exist.")
        return documents

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                documents.append({
                    'filename': filename,
                    'content': content
                })
                app.logger.debug(f"Loaded document: {filename}")
            except Exception as e:
                app.logger.error(f"Error reading {filepath}: {e}")
    if not documents:
        app.logger.warning("No .txt files were found in the folder.")
    return documents

class Indexer:
    def __init__(self, documents):
        """
        Create a TF-IDF index from a list of document dictionaries.
        Each document in documents should have 'filename' and 'content'.
        """
        self.documents = documents
        corpus = [doc['content'] for doc in documents]
        if corpus:
            try:
                self.vectorizer = TfidfVectorizer(stop_words='english')
                self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
                app.logger.debug("TF-IDF index created successfully.")
            except Exception as e:
                app.logger.error(f"Error creating TF-IDF index: {e}")
                self.vectorizer = None
                self.tfidf_matrix = None
        else:
            self.vectorizer = None
            self.tfidf_matrix = None
            app.logger.warning("Empty corpus provided to the indexer.")

    def search(self, query, top_k=10):
        if self.vectorizer is None or self.tfidf_matrix is None:
            app.logger.warning("TF-IDF index is missing; cannot perform search.")
            return []
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append({
                    'filename': self.documents[idx]['filename'],
                    'content': self.documents[idx]['content'],
                    'similarity': float(similarities[idx])
                })
            return results
        except Exception as e:
            app.logger.error(f"Error during search: {e}")
            return []

class Chatbot:
    def __init__(self, indexer):
        self.indexer = indexer
        self.conversation_history = []
    
    def process_query(self, query):
        self.conversation_history.append({"user": query})
        results = self.indexer.search(query)
        if results:
            response = {
                "message": "Relevant documents found:",
                "results": results 
            }
        else:
            response = {"message": "I couldn't find any information matching your query. Could you please clarify?"}
        self.conversation_history.append({"bot": response})
        return response

DOCUMENTS_DIR = './course_materials'
documents = load_documents(DOCUMENTS_DIR)
indexer = Indexer(documents)
chatbot = Chatbot(indexer)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.form if request.form.get("query") else request.get_json(force=True)
        app.logger.debug(f"Received data: {data}")
        query = data.get("query", "")
        if not query:
            msg = "Query field is missing"
            app.logger.error(msg)
            return jsonify({"error": msg}), 400

        response = chatbot.process_query(query)
        if request.form.get("query"):
            return render_template("index.html", query=query, response=response)
        return jsonify({"response": response})
    except Exception as e:
        app.logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)