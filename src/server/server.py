from flask import Flask, flash, redirect, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import tempfile

app = Flask(__name__)

# Document
current_document = None

# Setup the RAG chat engine


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return render_template('chat.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_pdf():
    # Check if the post request has the file part
    if 'pdf_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['pdf_file']
    
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and file.content_type == 'application/pdf':
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))
    else:
        flash('Invalid file format')
        return redirect(request.url)


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg

    # check if text contains references

    # update vector store if it does

    # get response from RAG

    return "Go away!"

