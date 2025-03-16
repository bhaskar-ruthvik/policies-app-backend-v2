from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
from utils.utils import formatFlowchartType, formatParagraphType, getCategoryOfInput, getResponseFromLLM, retrieve_closest_document_policies, retrieve_closest_document_legal
import time


app = Flask(__name__)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=key)

@app.route('/',methods=["POST"])
def index():
    if request.method == "POST": 
        # curr = time.time()
        ip = request.form.get("body")
        state = request.form.get("state")
        type_of_q = request.form.get("type")
        if not type:
            type_of_q = "Policies"
        
        # end = time.time() 
        # print("Read arguments complete. Time taken: ", end-curr)
        # curr = time.time()
        if not state:
            state = "Central"
        
        # Determine category of the input
        cat = getCategoryOfInput(ip,api_key)
        # end = time.time()
        # print("Got Category of Input. Time Taken: ", end-curr)
        # curr = time.time()
        if type_of_q == "Policies":
            retrieved_docs = retrieve_closest_document_policies(ip, state)
        else:
            retrieved_docs = retrieve_closest_document_legal(ip,state)
        context=  "\n\n".join([doc.page_content for doc in retrieved_docs])
        # end = time.time()
        # print("Retrieved Relevant Documents. Time Taken:", end-curr)
        # curr = time.time()
        content = getResponseFromLLM(ip,context ,cat, api_key)
        
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
        # end = time.time()
        # print("Response Formed. Time Taken: ", end-curr)
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
