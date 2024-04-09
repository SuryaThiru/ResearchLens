import fitz  # PyMuPDF
import re
from rapidfuzz import fuzz
from tqdm import tqdm
import requests
import logging
from semanticscholar import SemanticScholar


def _get_extract_blocks_from_doc(extract, doc, fuzzy_threshold=0.9, min_text_len=10):
    """
    Extracts text blocks from a document that match a given extract.

    Args:
        extract (str): The text to match against the text blocks.
        doc: The fitz document object representing the document to extract from.
        fuzzy_threshold (float, optional): The minimum fuzzy match score required for a block to be considered a match. Defaults to 0.9.
        min_text_len (int, optional): The minimum length of the text block required for it to be considered a match. Defaults to 10.

    Returns:
        list: A list of tuples containing the matching text blocks, their page numbers, and their match scores.

    """
    matches = []

    for page_num in tqdm(
        range(len(doc)), leave=False, desc="Extracting matching text from document"
    ):
        page = doc.load_page(page_num)  # load the current page
        text_blocks = (
            page.get_text_blocks()
        )  # get a list of text blocks on the current page
        for block in text_blocks:
            text = block[4]

            match_score = fuzz.partial_ratio(extract, text)

            if match_score >= fuzzy_threshold and len(text) >= min_text_len:
                matches.append((block, page_num, match_score))

    matches = sorted(matches, key=lambda x: x[2], reverse=True)

    return matches


def _get_links_in_block(doc, block, page):
    # get the matched region
    bbox = fitz.Rect(block[:4])

    # get the citation links
    matched_links = []

    for link in doc[page].get_links():
        if link["kind"] == 4:  # internal links
            link_bbox = link["from"]
            if bbox.intersects(link_bbox):
                link["from_page"] = page
                matched_links.append(link)

    return matched_links


def _get_references_from_matches(doc, matches, extract):
    """
    Given text block matches of an extract in a document,
    extracts the references from the matched text blocks.

    Function first extracts links from the matched blocks,
    then uses the citation numbers (not names as of now)
    to extract the references from the bibliography
    section of the paper.

    Any citations present in matches that are not present
    in the extract are filtered out.
    """

    # Get links in the matches
    matched_links = []

    for match in matches:
        matched_links.extend(_get_links_in_block(doc, match[0], match[1]))

    # filter the links and update it with citation number
    matched_links_filtered = []

    for link in matched_links:
        # keep only citations, not equations and figures
        if not link["nameddest"].startswith("cite."):
            continue

        citation_num = doc[link["from_page"]].get_text("text", clip=link["from"])
        citation_num = re.findall(r"\d+", citation_num)

        if "page" not in link:
            logging.warning(f"Page link not found for {citation_num}")
            continue

        if len(citation_num) == 0:
            continue

        citation_num = citation_num[0]

        if citation_num not in extract:
            continue

        link["citation_number"] = citation_num
        matched_links_filtered.append(link)

    # Remove duplicate citations
    unique_matches = {}

    for match in matched_links_filtered:
        citation_num = match["citation_number"]

        # already exist, duplicate - keep if a link has more attributes than an existing one
        if citation_num in unique_matches and len(match.keys()) < len(
            unique_matches[citation_num]
        ):
            continue

        # doesn't exist
        unique_matches[citation_num] = match

    matched_links_filtered = list(unique_matches.values())

    # Get the reference corresponding to each citation
    matched_references = []

    for link in matched_links_filtered:
        linked_page = doc.load_page(link["page"])
        text_blocks = linked_page.get_text("blocks")
        citation_num = link["citation_number"]
        num_pat = r"\b" + citation_num + r"\b"

        for text in text_blocks:
            # citation number should be present in the initial section of the reference
            if re.search(num_pat, text[4][:15]):
                matched_references.append(text[4].strip())

    # Remove line breaks
    matched_references = list(map(lambda x: x.replace("\n", " "), matched_references))
    matched_references = list(map(lambda x: x.replace("- ", ""), matched_references))

    return matched_references


def _get_title_from_reftext(
    reftext,
    anystyle_url="http://localhost:4567/parse",
    request_timeout=3,
    min_title_len=15,
):
    """
    Extracts the title from the given reference text using the Anystyle API.

    Args:
        reftext (str): The reference text from which to extract the title.
        min_title_len (int, optional): The minimum length of the extracted title. Defaults to 15.
        anystyle_url (str, optional): The URL of the Anystyle API. Defaults to 'http://localhost:4567/parse'.

    Returns:
        str: The extracted title.

    Raises:
        AssertionError: If the extracted title is shorter than the minimum title length.

    """
    reftext = reftext.encode("utf-8")
    response = requests.post(
        anystyle_url,
        headers={"Content-Type": "text/plain"},
        data=reftext,
        timeout=request_timeout,
    )
    parsed_data = response.json()

    parsed_data = parsed_data[0]
    title = parsed_data.get("title")
    if title is None:
        return title

    title = " ".join(title)

    assert len(title) >= min_title_len

    return title


def _get_metadata_of_references(
    references, anystyle_url, semantic_scholar_api_key, request_timeout
):
    """
    Retrieves metadata for a list of text of reference.

    Args:
        references (list): A list of reference texts.
        anystyle_url (str): The URL of the Anystyle service.
        semantic_scholar_api_key (str): The API key for the Semantic Scholar service.
        request_timeout (int): The timeout for HTTP requests.

    Returns:
        list: A list of dictionaries containing the metadata for each reference.
    """
    # get title from reference text
    references_title = []
    for reftext in tqdm(references, desc="Extracting titles from references"):
        title = _get_title_from_reftext(reftext, anystyle_url, request_timeout)
        references_title.append(title)

    # search for the metadata using the paper title
    sch = SemanticScholar(api_key=semantic_scholar_api_key, timeout=request_timeout)

    references_meta = []
    for ref in tqdm(references_title, desc="Fetching metadata"):
        if ref is None:
            references_meta.append(None)
            continue

        results = sch.search_paper(
            ref, limit=1, fields=["title", "paperId", "externalIds", "openAccessPdf"]
        )

        if results.total == 0:
            logging.warning(f'No metadata found for paper "{ref}"')
            references_meta.append(None)
            continue

        meta = results[0]
        references_meta.append(meta.raw_data)

    # get the pdf URL of each paper
    for meta in references_meta:
        if meta is None:
            continue

        if meta["openAccessPdf"] is not None:
            meta["pdf_url"] = meta["openAccessPdf"]["url"]
        elif "ArXiv" in meta["externalIds"]:
            meta["pdf_url"] = f"https://arxiv.org/pdf/{meta['externalIds']['ArXiv']}"
        else:
            meta["pdf_url"] = None

    return references_meta


def extract_references_from_doc_extract(
    doc,
    extract,
    semantic_scholar_api_key,
    anystyle_url="http://localhost:4567/parse",
    request_timeout=10,
    fuzzy_threshold=92,
    min_matched_text_len=10,
):
    if isinstance(doc, str):
        doc = fitz.open(doc)
    elif isinstance(doc, fitz.Document):
        pass
    else:
        raise ValueError("doc should be a file path or a fitz Document object")

    matched_blocks = _get_extract_blocks_from_doc(
        extract,
        doc,
        fuzzy_threshold,
        min_matched_text_len,
    )
    references = _get_references_from_matches(doc, matched_blocks, extract)

    # log extracted citations
    for ref in references:
        logging.info(f"Extracted reference: {ref}")

    metadata = _get_metadata_of_references(
        references, anystyle_url, semantic_scholar_api_key, request_timeout
    )

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datadir = "/home/surya/NEU/CS5100 FAI/Project/pdfreader/"
    file = datadir + "test2.pdf"
    doc = fitz.open(file)
    extract = """
    Bradley-Terry model [5], then fine-tune a language model to maximize the given reward using
reinforcement learning algorithms, commonly REINFORCE [45], proximal policy optimization
(PPO; [37]), or variants [32]. A closely-related line of work leverages LLMs fine-tuned for instruction
following with human feedback to generate additional synthetic preference data for targeted attributes
such as safety or harmlessness [2], using only weak supervision from humans in the form of a
text rubric for the LLM’s annotations. These methods represent a convergence of two bodies of
work: one body of work on training language models with reinforcement learning for a variety
of objectives [33, 27, 46] and another body of work on general methods for learning from human
preferences [12, 19]. Despite the appeal of using relative human preferences, fine-tuning large
language models with reinforcement learning remains a major practical challenge; this work provides
a theoretically-justified approach to optimizing relative preferences without RL.

Describe the Bradley-Terry model.
    """
    metadata = extract_references_from_doc_extract(
        doc,
        extract,
        anystyle_url="https://anystyle-webapp.azurewebsites.net/parse",
        semantic_scholar_api_key="WWxz8zHVUm6DWzkmw6ZSd3eA94kWbbX46Zl5jR11",
    )

    print("Extracted titles:")
    for meta in metadata:
        print(meta["title"])

    print("Extracted pdf URLs:")
    for meta in metadata:
        print(meta["pdf_url"])
