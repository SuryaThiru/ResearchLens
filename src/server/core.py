from src.refextract.extract import extract_references_from_doc_extract


def preprocess_chat(doc, text):
    metadata = extract_references_from_doc_extract(
        doc,
        text,
        anystyle_url="https://anystyle-webapp.azurewebsites.net/parse",
        semantic_scholar_api_key="WWxz8zHVUm6DWzkmw6ZSd3eA94kWbbX46Zl5jR11",
    )
