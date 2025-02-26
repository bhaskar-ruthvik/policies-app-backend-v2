from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import firebase_admin
from firebase_admin import credentials, firestore
import openai
import numpy as np

cred = credentials.Certificate("serviceAccountKey.json")  # Update with your service key
firebase_admin.initialize_app(cred)
db = firestore.client()

key = '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' #this is the new key
embeddings = OpenAIEmbeddings(api_key=key)

def get_embedding(text):
    return embeddings.embed_query(text)

def cosine_similarity(vec1, vec2):
    #see if theres a better method for this
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_closest_document(query,state):
    """Finds the closest matching vector from Firestore and returns its metadata."""
    query_embedding = get_embedding(query)
    closest_doc = None
    max_similarity = -1

    # Access Firestore collection
    files_ref = db.collection("test-files").where("state", "==", state)
    the_files = files_ref.stream() #store all states
    #print(states_docs)
   
    files = sorted([{"content": data.to_dict().get("text",""), "similarity": cosine_similarity(query_embedding,data.to_dict()["vector"])} for data in the_files], key = lambda data: data['similarity'],reverse = True)
   
    # for file_doc in the_files: #for each file in the file docs
    #     #instead of going file by file find a way to retrieve by state using dictionary
    #     data = file_doc.to_dict() 
    #     # print('searching files')
    #     if "vector" in data:
    #         #doc_vector = data["vector"]
    #         doc_vector = np.array(data["vector"], dtype=np.float32)
    #         similarity = cosine_similarity(query_embedding, doc_vector)

    #         print(f"New doc found with similarity: {similarity}")

    #         if similarity > max_similarity:
    #                 max_similarity = similarity
    #                 closest_doc = {
    #                     "content": data.get("text", ""),
    #                     "state": data.get("state", "")                   
    #                 }
    
    return files[:5] if (len(files) > 5) else files

query = "Farmer Financial assistance"
state = "Telangana"
matched_docs = retrieve_closest_document(query,state)
print([x['content'][:500] for x in matched_docs])
# if matched_doc:
#     print("Matched Document Content:", matched_doc["content"][:500], "...\n")
#     print("Metadata:", matched_doc["state"], "\n")
# else:
#     print("No matching document found.")

