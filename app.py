from flask import Flask, jsonify, request
import os
from dotenv import load_dotenv
from utils.utils import formatFlowchartType, formatParagraphType, getCategoryOfInput, getResponseFromLLM


app = Flask(__name__)
model_name = "gpt-4o"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=key)

@app.route('/',methods=["POST"])
def index():
    if request.method == "POST": 
        ip = request.form.get("body")
        
        # Determine category of the input
        cat = getCategoryOfInput(ip,api_key)
        
        # # Retrieve relevant documents for context
        # retrieved_documents = retrieveDocuments(ip)
        # context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        
        # Generate the response with the retrieved context
        content = getResponseFromLLM(ip, cat, api_key)
        
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
