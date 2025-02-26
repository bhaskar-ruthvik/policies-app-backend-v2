# from langchain.embeddings import OpenAIEmbeddings
import os
import google.cloud
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from openai import OpenAI
import numpy as np

cred = credentials.Certificate("serviceAccountKey.json")  # Update with your service key
firebase_admin.initialize_app(cred)
db = firestore.client()


key = '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' #this is the new key
# embeddings = OpenAIEmbeddings(api_key=key)
client = OpenAI(api_key = key)

def get_embedding(text):
    return client.embeddings.create(
        input = "How to apply for ration card",
        model = "text-embedding-3-small"
    ).data[0].embedding

def cosine_similarity(vec1, vec2):
    #see if theres a better method for this
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_closest_document(query):
    """Finds the closest matching vector from Firestore and returns its metadata."""
    # query_embedding = get_embedding(query)
    closest_doc = None
    max_similarity = 0
    docs = db.collection("test-files")
    embedding = get_embedding("Hello")
    # print(embedding)
    resp = docs.find_nearest(
        vector_field  = "vector",
        query_vector = Vector(embedding),
        distance_measure = DistanceMeasure.EUCLIDEAN,
        limit = 3
    )
    for docu in resp.stream():
        print(docu.to_dict())
    #print(db.collection("states").stream())
    # Access Firestore collection
    # flag = 0
    # for doc in docs:
    #     if flag == 0:
    #         print(doc.to_dict())
    #         flag = 1


    
    # for state_doc in states_docs: #for each state folder
    #     files_ref = state_doc.reference.collection("files") #in files collection
    #     files_docs = files_ref.stream() #store all files
     
    #     for file_doc in files_docs: #for each file in the file docs
    #         data = file_doc.to_dict() 
    #         print('searching files')
    #         if "vector" in data and "metadata" in data:
    #             #doc_vector = data["vector"]
    #             doc_vector = np.array(data["vector"], dtype=np.float32)
    #             similarity = cosine_similarity(query_embedding, doc_vector)

    #             print(f"New doc found with similarity: {similarity}")

    #             if similarity > max_similarity:
    #                 max_similarity = similarity
    #                 closest_doc = {
    #                     "content": data.get("text", ""),
    #                     "metadata": data["metadata"]
    #                 }
                    

    return 0

query = "Scheme for Grant of Additional Scholarship to the Students of Other Backward Classes of Andaman?"
matched_doc = retrieve_closest_document(query)

if matched_doc:
    print("Matched Document Content:", matched_doc["content"][:500], "...\n")
    print("Metadata:", matched_doc["metadata"], "\n")
else:
    pass
    #print("No matching document found.")
