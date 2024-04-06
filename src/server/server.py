from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile


app = Flask(__name__)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.content_type == 'application/pdf':
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Secure the filename
        filename = secure_filename(file.filename)
        # Save file to the temporary directory
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)

        # Put the pdf in a vectorstore

        return jsonify({"message": "File uploaded successfully"}), 200
    else:
        return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    # check if text contains references

    # update vector store if it does

    # get response from RAG

    return "Go away!"

