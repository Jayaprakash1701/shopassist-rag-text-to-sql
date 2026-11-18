import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os
import re

base_dir = Path(__file__).parent
load_dotenv(base_dir / ".env")

faq_path = base_dir / "resources" / "faq.csv"
chroma_client = chromadb.Client()
collection_name_faq = 'faqs'
groq_client = Groq()

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
    )


def ingest_faq_data(path):
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.get_or_create_collection(
            name=collection_name_faq,
            embedding_function = ef 
        )

        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f'id_{i}' for i in range(len(docs))]

        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )

        print(f"FAQ data ingested into ChromaDB collection '{collection_name_faq}'")
    else:
        print(f"Collection '{collection_name_faq}' already exists. Skipping ingestion.")

def get_relevant_qa(query):
    collection = chroma_client.get_collection(name=collection_name_faq)
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    return results

def faq_chain(query):
    results = get_relevant_qa(query)

    context = "".join(r.get('answer') for r in results['metadatas'][0])
    answer = generate_answer(query, context)
    return answer

def faq_chain_with_context(query):
    results = get_relevant_qa(query)

    context = "".join(r.get('answer') for r in results['metadatas'][0])
    answer = generate_answer(query, context)
    return answer, context
import time
from groq import RateLimitError

def call_groq_with_retry(client, **kwargs):
    for attempt in range(10):
        try:
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, Exception) as e:
            if "429" in str(e) or "RateLimit" in str(e) or isinstance(e, RateLimitError):
                time.sleep(5.0)
            else:
                raise e
    return client.chat.completions.create(**kwargs)

def generate_answer(query, context):

    prompt = f'''given the following question and answer pairs, answer the question in a concise manner. If the answer is not present, say "I don't know".do not make up an answer.
    Question: {query}

    CONTEXT : {context}
    ''' 
    chat_completion = call_groq_with_retry(
        groq_client,
        messages = [
            {
                "role" : "user",
                "content" : prompt,
            }
        ],
        model = os.environ['GROQ_MODEL'],
    )
    ans = chat_completion.choices[0].message.content
    ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.DOTALL).strip()
    return ans


if __name__ == "__main__":
    ingest_faq_data(faq_path)
    query = "is upi payment supported?"
    #results = get_relevant_qa(query)
    #print("Query Results:", results)
    answer = faq_chain(query)
    print("Answer:", answer)