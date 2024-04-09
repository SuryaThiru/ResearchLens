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
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
import cohere


def is_exist_in_vector_store_index(index, pdf_file):
    for node in index.docstore.docs.values():
        if node.metadata["file_path"] == pdf_file:
            return True
    return False


def update_vector_store_index(index, pdf_file):
    if is_exist_in_vector_store_index(index, pdf_file):
        logging.info(f"Document {pdf_file} already present in the index. Skipping.")
        return

    documents = SimpleDirectoryReader(input_files=[pdf_file]).load_data()

    for doc in documents:
        index.insert(doc)

    logging.info(f"Document {pdf_file} added to the index.")


def setup_chat_engine(directory):
    logging.info(f"Loading documents from {directory} directory")
    documents = SimpleDirectoryReader(directory).load_data()
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

    embed_model = CohereEmbedding(
        cohere_api_key="vORtxj32na8zl2ceIbxH1c5tNziAVWDdAy2x3sbX",
        model_name="embed-english-v3.0",  # Supports all Cohere embed models
        input_type="search_query",  # Required for v3 models
    )

    logging.info("Loading LLM model")
    llm_model = Cohere(
        model="command-r",
        api_key="vORtxj32na8zl2ceIbxH1c5tNziAVWDdAy2x3sbX",
        temperature=0.1,
    )

    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter
    Settings.llm = llm_model

    logging.info("Building vector store index")
    index = VectorStoreIndex.from_documents(
        documents, embed_model=embed_model, transformations=[text_splitter]
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    logging.info("Creating chat engine")
    chat_engine = index.as_chat_engine(
        # chat_mode="condense_plus_context",
        chat_mode="context",
        memory=memory,
        llm=llm_model,
        context_prompt=(
            "You are a chatbot, able to have normal interactions, as well as explaining research papers."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        ),
        verbose=True,
    )

    return chat_engine, index


def _get_full_text(file_path):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    full_doc_text = "\n".join([doc.text for doc in documents])
    return full_doc_text


def extract_with_llm(original_question, file_path, max_tokens=1000):
    full_doc_text = _get_full_text(file_path)

    prompt = f"""
Given the following question about a paper from the user, please provide a brief compilation of texts from the referring paper that can help answer the question.

Question:
{original_question}

Referring Paper Content:
{full_doc_text}
"""

    co = cohere.Client("vORtxj32na8zl2ceIbxH1c5tNziAVWDdAy2x3sbX")
    response = co.chat(
        message=prompt,
        model="command-r",
        temperature=0.1,
        max_tokens=max_tokens,
    )

    answer = response.text
    return answer


def improve_prompt_with_citing_context(
    original_question, citing_papers, max_tokens=1000
):
    citation_context = "Citation Context:\n"
    for title, file_path in citing_papers:
        logging.info(f"Adding context of {title} to the prompt.")
        full_text = extract_with_llm(
            original_question, file_path, max_tokens=max_tokens
        )
        citation_context += f"Paper: {title}\n{full_text}\n"

    prompt = f"{citation_context}\nQuestion: {original_question}"
    return prompt


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
