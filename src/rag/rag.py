from src.refextract.extract import extract_references_from_doc_extract
import logging
import fitz
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer


def setup_chat_engine(directory):
    logging.info(f"Loading documents from {directory} directory")
    documents = SimpleDirectoryReader(directory).load_data()
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    logging.info("Loading LLM model")
    cohere_model = Cohere(api_key="vORtxj32na8zl2ceIbxH1c5tNziAVWDdAy2x3sbX")

    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.llm = cohere_model

    logging.info("Building vector store index")
    index = VectorStoreIndex.from_documents(documents)

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    logging.info("Creating chat engine")
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=cohere_model,
        context_prompt=(
            "You are a chatbot, able to have normal interactions, as well as explaining research papers."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        ),
        verbose=True,
    )

    return chat_engine, index


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

    directory = (
        "C:/Users/bayan/Desktop/Github/ResearchLens/src/refextract/pdf_metadata/"
    )
    ensure_pdfs_are_downloaded(metadata, directory)
    chat_engine(directory)
