# retrieve relevant docs and add them to the prompts i added


from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
#from utils.utils import formatFlowchartType, formatParagraphType, getCategoryOfInput, getResponseFromLLM
from utils import formatFlowchartType, formatParagraphType, getCategoryOfInput, getResponseFromLLM


app = Flask(__name__)


load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key)

def get_embedding(text):
    return embeddings.embed_query(text)

def cosine_similarity(vec1, vec2):
    #see if theres a better method for this
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieveDocuments(query,state):
    """Finds the closest matching vector from Firestore and returns its metadata."""
    query_embedding = get_embedding(query)
    closest_doc = None
    max_similarity = -1

    # Access Firestore collection
    files_ref = db.collection("test-files").where("state", "==", state)
    the_files = files_ref.stream() #store all states

    for file_doc in the_files: #for each file in the file docs
        #instead of going file by file find a way to retrieve by state using dictionary
        data = file_doc.to_dict() 
        print('searching files')
        if "vector" in data:
            doc_vector = np.array(data["vector"], dtype=np.float32)
            similarity = cosine_similarity(query_embedding, doc_vector)

            print(f"New doc found with similarity: {similarity}")

            if similarity > max_similarity:
                    max_similarity = similarity
                    closest_doc = {
                        "content": data.get("text", ""),
                        "state": data.get("state", "")                   
                    }

    return closest_doc

@app.route('/',methods=["POST"])
def index():
    if request.method == "POST": 
        ip = request.form.get("body")
        state = request.form.get("body")
        
        # Determine category of the input
        cat = getCategoryOfInput(ip,api_key)
        
        # # Retrieve relevant documents for context
        retrieved_document = retrieveDocuments(ip,state)
        # context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        
        # Generate the response with the retrieved context
        content = getResponseFromLLM(ip, cat, api_key, retrieved_document)
        
        if cat == "Informative Paragraph Question":
            headings, slugs = formatParagraphType(content)
            body = {
                "headings": headings,
                "slugs": slugs
            }
        elif cat == "Procedure-Based Question":
            body = formatFlowchartType(content)
        else:
            [val, cont] = content.split("\n\n", 1)
            body = {
                "value": val,
                "content": cont
            }

        data = {
            "type": cat,
            "body": body
        }
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
