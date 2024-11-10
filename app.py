from flask import Flask, render_template, request

from ecomm.data_ingestion import ingest_data

from ecomm.retrieval_generation import create_conversational_chain

# Ingest data and create vector store
vstore, insert_ids = ingest_data()
chain = create_conversational_chain(vstore)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST", "GET"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]
        input = msg

        result = chain.invoke(
            {"input": input},
            config={
                "configurable": {"session_id": "sha"}
            },
        )["answer"]

        return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
