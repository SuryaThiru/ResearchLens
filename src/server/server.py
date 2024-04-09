import os
import logging
import cohere
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.serving import is_running_from_reloader

import sys

from werkzeug.utils import secure_filename

sys.path.append(".")

from src.rag import (
    setup_chat_engine,
    ensure_pdfs_are_downloaded,
    update_vector_store_index,
    improve_prompt_with_citing_context
)
from src.refextract import extract_references_from_doc_extract


app = Flask(__name__)
app.logger.setLevel(level=logging.INFO)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploads"
UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Document
current_document = None

# Setup the RAG chat engine
# chat_engine, vectordb = setup_chat_engine(UPLOAD_FOLDER)
from werkzeug.serving import is_running_from_reloader

# Setup the RAG chat engine
chat_engine, vectordb = None, None
if not is_running_from_reloader():
    chat_engine, vectordb = setup_chat_engine(UPLOAD_FOLDER)


@app.route("/")
def index():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return render_template("chat.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    # Check if the post request has the file part
    if "pdf_file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["pdf_file"]

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file and file.content_type == "application/pdf":
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        update_vector_store_index(vectordb, filepath)
        return redirect(url_for("index"))
    else:
        flash("Invalid file format")
        return redirect(request.url)


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    filename = request.form.get("filename")
    input = msg

    app.logger.info(f"Received message: {msg} for {filename}")

    if filename is None or filename == "":
        response = chat_engine.chat(input)
        return response.response

    # check if text contains references
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    metadata = extract_references_from_doc_extract(
        filepath,
        input,
        anystyle_url="https://anystyle-webapp.azurewebsites.net/parse",
        semantic_scholar_api_key="WWxz8zHVUm6DWzkmw6ZSd3eA94kWbbX46Zl5jR11",
        request_timeout=20,
        fuzzy_threshold=80,
    )
    metadata = [m for m in metadata if m is not None]
    pdfs = ensure_pdfs_are_downloaded(metadata, UPLOAD_FOLDER)
    app.logger.info(f"Using PDFs: {' '.join(map(str, pdfs))}")

    referenced_papers = [m['title'] for m in metadata if m is not None]

    # update vector store if it does not exist
    # for pdf in pdfs:
    #     update_vector_store_index(vectordb, pdf)

    # citing papers
    citing_papers = [(title, pdf) for title, pdf in zip(referenced_papers, pdfs)]

    improved_prompt = improve_prompt_with_citing_context(input, citing_papers, main_paper=filepath)
    app.logger.info(f"Improved prompt: {improved_prompt}")

    # get response from RAG
    # response = chat_engine.chat(improved_prompt)
    # response = response.response
    co = cohere.Client("MImk1OQjETu9s31ZWNlLtDfrn6bNPG9AJXpTrbfu")
    response = co.chat(
        message=improved_prompt,
        model="command-r",
        temperature=0.1,
        max_tokens=2048,
    )
    response = response.text

    # app.logger.info(f"Retrieved {len(response.source_nodes)} Source nodes")
    # for src in response.source_nodes:
    #     app.logger.info(src)

    # Add references to the response
    referenced_papers = "\n".join(referenced_papers)

    reference_template = f"Referenced papers:\n{referenced_papers}"
    response = response + "\n\n" + reference_template

    return response
