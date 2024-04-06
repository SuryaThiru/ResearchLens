from src.refextract.utils import download_pdf
from src.refextract.extract import extract_references_from_doc_extract
import os
import logging
import fitz
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from IPython.display import Markdown, display
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer


def ensure_pdfs_are_downloaded(metadata, directory):
    directory = Path(directory)

    for meta in metadata:
        if meta and "pdf_url" in meta and meta["pdf_url"]:
            safe_title = "".join(c for c in meta['title'] if c.isalnum() or c in " _-").rstrip()
            pdf_filename = directory / f"{safe_title}.pdf"
            if not pdf_filename.exists():
                download_pdf(meta["pdf_url"], pdf_filename)
            else:
                logging.info(f"{pdf_filename} already exists.")


def chat_engine(directory):
    documents = SimpleDirectoryReader(directory).load_data()
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
    cohere_model = Cohere(api_key="vORtxj32na8zl2ceIbxH1c5tNziAVWDdAy2x3sbX")
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.llm = cohere_model

    index = VectorStoreIndex.from_documents(documents)

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=cohere_model,
        context_prompt=(
            "You are a chatbot, able to have normal interactions, as well as explaining research papers."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."),
        verbose=True,
    )

    # chat_engine = index.as_chat_engine(
    # chat_mode="condense_plus_context",
    # context_prompt=(
    #     "You are a chatbot, able to have normal interactions, as well as talk"
    #     " about an essay discussing Paul Grahams life."
    #     "Here are the relevant documents for the context:\n"
    #     "{context_str}"
    #     "\nInstruction: Based on the above documents, provide a detailed answer for the user question below."),
    # )

    response = chat_engine.chat("""what are the 6 main duties to support the delivery of functionalities  𝑃1, 𝑃2, 𝑅1, and 𝑅2 """.strip())

    response.source_nodes
    print(response)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datadir = "src/refextract/pdf_metadata/"
    file = datadir + "2307.06435-2.pdf"
    doc = fitz.open(file)
    extract = """
The application of LLMs in the field of medicine
is reshaping healthcare delivery and research. For example,
LLMs are increasingly used in clinical decision support systems to provide physicians with evidence-based treatment
recommendations [425], [426], [427]. By analyzing patient
data and medical literature, they can help identify potential
diagnoses, suggest appropriate tests, and recommend optimal
treatment strategies. Moreover, LLMs can also enhance patient
interactions with healthcare systems; e.g., they can be used
in chatbot applications [428], [429], [430] to answer patient
queries about symptoms or medications, schedule appointments, and even provide essential health advice. For medical
research, LLMs are used to extract and filter information from
a considerable amount of medical literature, identify relevant
studies, summarize findings, and even predict future research
trends [431], [432], [433].
    """
    metadata = extract_references_from_doc_extract(
        doc,
        extract,
        anystyle_url="https://anystyle-webapp.azurewebsites.net/parse",
        semantic_scholar_api_key="WWxz8zHVUm6DWzkmw6ZSd3eA94kWbbX46Zl5jR11",
    )

    directory = "C:/Users/bayan/Desktop/Github/ResearchLens/src/refextract/pdf_metadata/"
    ensure_pdfs_are_downloaded(metadata, directory)
    chat_engine(directory)

