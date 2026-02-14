import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_document(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def split_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def create_vector_store(chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


def retrieve(query, vectorizer, vectors, chunks, top_k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, vectors)
    top_indices = np.argsort(similarities[0])[-top_k:]
    return [chunks[i] for i in top_indices]


def generate_response(context, question):
    prompt = f"""
You are a customer support assistant.
Answer using only past ticket resolutions.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def main():
    document = load_document("tickets.txt")
    chunks = split_text(document)
    vectorizer, vectors = create_vector_store(chunks)

    while True:
        question = input("\nAsk Support Question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        relevant_chunks = retrieve(question, vectorizer, vectors, chunks)
        context = "\n\n".join(relevant_chunks)

        answer = generate_response(context, question)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()