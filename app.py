import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ Hugging Face token not found. Please set HF_TOKEN.")
os.environ["HF_TOKEN"] = hf_token

# Hugging Face embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Books dataset
books = pd.read_csv("Models/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("static/cover-not-found.jpg") + "&fife=w800"
for col in ["joy", "surprise", "anger", "fear", "sadness"]:
    if col not in books.columns:
        books[col] = 0.0

# Load docs
raw_documents = TextLoader("Models/tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Vector DB
db_books = Chroma.from_documents(documents, embedding=embedding)

# --- Core Logic (unchanged) ---
def retrieve_semantic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append({"thumbnail": row["large_thumbnail"], "caption": caption})
    return results

# --- Flask App ---
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    categories = ["All"] + sorted(books["simple_categories"].unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    results = []

    if request.method == "POST":
        query = request.form.get("query")
        category = request.form.get("category")
        tone = request.form.get("tone")
        results = recommend_books(query, category, tone)

    return render_template("index.html", categories=categories, tones=tones, results=results)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

