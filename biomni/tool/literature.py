import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from io import BytesIO
from urllib.parse import urljoin

import PyPDF2
import requests
from bs4 import BeautifulSoup
from googlesearch import search

from biomni import credentials

logger = logging.getLogger(__name__)


def _search_unavailable(source: str, why: str, instead: str) -> str:
    """What a search tool answers when it could not search: never an empty result.

    An empty string, or "no papers found", reads as "nothing exists on this",
    and the agent carried on as if it had searched. This says the search did
    not happen, why, and where to look instead - and logs it for the operator.
    """
    logger.warning("%s search unavailable: %s", source, why)
    return (
        f"{source} search is unavailable: {why}. "
        f"This is not an empty result: what {source} holds on this query is unknown. {instead}"
    )


def _get_with_retries(
    send: Callable[[], requests.Response],
    describe: Callable[[requests.Response], str],
    waits: tuple[float, ...],
    retry_statuses: frozenset[int],
) -> tuple[requests.Response | None, str, bool]:
    """``send()``, tried again after each of ``waits`` while it fails in a way that can pass.

    Returns the answer - the first 200, or the first refusal not worth retrying -
    and, unless it is a 200, why it holds no results. Once every try has failed,
    the answer is ``None`` and the last item is True.
    """
    why = ""
    for tries, wait in enumerate((0.0, *waits), start=1):
        time.sleep(wait)
        try:
            response = send()
        except requests.RequestException as e:
            why = f"the request failed ({type(e).__name__}: {e})"
            continue
        if response.status_code == requests.codes.ok:
            return response, "", False
        why = describe(response)
        if response.status_code not in retry_statuses:
            return response, _on_try(why, tries, waits), False
    return None, _on_try(why, len(waits) + 1, waits), True


def _on_try(why: str, tries: int, waits: tuple[float, ...]) -> str:
    """``why``, saying which try it came from when there were several."""
    if tries == 1:
        return why
    return f"{why}, on the last of {tries} tries over {sum(waits[: tries - 1]):.0f} seconds"


def fetch_supplementary_info_from_doi(doi: str, output_dir: str = "supplementary_info"):
    """Fetches supplementary information for a paper given its DOI and returns a research log.

    Args:
        doi: The paper DOI.
        output_dir: Directory to save supplementary files.

    Returns:
        dict: A dictionary containing a research log and the downloaded file paths.

    """
    research_log = []
    research_log.append(f"Starting process for DOI: {doi}")

    # CrossRef API to resolve DOI to a publisher page
    crossref_url = f"https://doi.org/{doi}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(crossref_url, headers=headers)

    if response.status_code != 200:
        log_message = f"Failed to resolve DOI: {doi}. Status Code: {response.status_code}"
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    publisher_url = response.url
    research_log.append(f"Resolved DOI to publisher page: {publisher_url}")

    # Fetch publisher page
    response = requests.get(publisher_url, headers=headers)
    if response.status_code != 200:
        log_message = f"Failed to access publisher page for DOI {doi}."
        research_log.append(log_message)
        return {"log": research_log, "files": []}

    # Parse page content
    soup = BeautifulSoup(response.content, "html.parser")
    supplementary_links = []

    # Look for supplementary materials by keywords or links
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        text = link.get_text().lower()
        if "supplementary" in text or "supplemental" in text or "appendix" in text:
            full_url = urljoin(publisher_url, href)
            supplementary_links.append(full_url)
            research_log.append(f"Found supplementary material link: {full_url}")

    if not supplementary_links:
        log_message = f"No supplementary materials found for DOI {doi}."
        research_log.append(log_message)
        return research_log

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    research_log.append(f"Created output directory: {output_dir}")

    # Download supplementary materials
    downloaded_files = []
    for link in supplementary_links:
        file_name = os.path.join(output_dir, link.split("/")[-1])
        file_response = requests.get(link, headers=headers)
        if file_response.status_code == 200:
            with open(file_name, "wb") as f:
                f.write(file_response.content)
            downloaded_files.append(file_name)
            research_log.append(f"Downloaded file: {file_name}")
        else:
            research_log.append(f"Failed to download file from {link}")

    if downloaded_files:
        research_log.append(f"Successfully downloaded {len(downloaded_files)} file(s).")
    else:
        research_log.append(f"No files could be downloaded for DOI {doi}.")

    return "\n".join(research_log)


# arXiv's API terms: one request at a time, and at most one every three seconds
# (https://info.arxiv.org/help/api/tou.html). A client that goes faster is
# throttled, and export.arxiv.org answers a throttled request with HTTP 406, not
# 429. Every query_arxiv call in the process - every chat's - shares this lock
# and clock, so the pace holds across calls, where a client per call kept none.
_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_HEADERS = {
    "User-Agent": "Biomni-AD (+https://github.com/Kaimen-Inc/Biomni-AD)",
    "Accept": "application/atom+xml",
}
_ARXIV_SPACING_S = 3.0
_ARXIV_TIMEOUT_S = (10, 30)
# Waits before each retry of a throttled or failing request: arXiv lifts its
# throttling after a while, and an outage can be brief.
_ARXIV_RETRY_WAITS_S = (5.0, 10.0, 20.0)
_ARXIV_THROTTLED = frozenset({406, 429})
_ARXIV_RETRY_STATUSES = _ARXIV_THROTTLED | {500, 502, 503, 504}
# How long arXiv is left alone once a request still fails after every retry:
# each later call would otherwise wait out the retries again, for the same
# answer, and add to the traffic that keeps a throttle on.
_ARXIV_REST_S = 120.0
_arxiv_lock = threading.Lock()
_arxiv_last_request = 0.0  # time.monotonic() when the last request ended
_arxiv_resting_until = 0.0  # time.monotonic() before which arXiv is not asked


def _arxiv_request(params: dict) -> requests.Response:
    """One arXiv API request, spaced from the previous one as arXiv asks."""
    global _arxiv_last_request
    wait = _arxiv_last_request + _ARXIV_SPACING_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.get(_ARXIV_API, params=params, headers=_ARXIV_HEADERS, timeout=_ARXIV_TIMEOUT_S)
    finally:
        _arxiv_last_request = time.monotonic()


def _describe_arxiv(response: requests.Response) -> str:
    why = f"arXiv answered HTTP {response.status_code}"
    if response.status_code in _ARXIV_THROTTLED:
        why += ", which it sends to clients it is throttling"
    return why


def _fetch_arxiv(params: dict) -> tuple[requests.Response | None, str]:
    """arXiv's answer to ``params`` and why it holds no results, as :func:`_get_with_retries` gives them."""
    global _arxiv_resting_until
    with _arxiv_lock:
        rest = _arxiv_resting_until - time.monotonic()
        if rest > 0:
            ago = _ARXIV_REST_S - rest
            when = "moments ago" if ago < 5 else f"{ago:.0f} seconds ago"
            why = f"arXiv was still failing after every retry {when}, so it is not asked again for {rest:.0f} seconds"
            return None, why
        response, why, exhausted = _get_with_retries(
            lambda: _arxiv_request(params), _describe_arxiv, _ARXIV_RETRY_WAITS_S, _ARXIV_RETRY_STATUSES
        )
        if exhausted:
            _arxiv_resting_until = time.monotonic() + _ARXIV_REST_S
    return response, why


def query_arxiv(query: str, max_papers: int = 10) -> str:
    """Query arXiv for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).

    Returns
    -------
    - str: The formatted search results or an error message.

    """
    import feedparser

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_papers,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    instead = "Search PubMed with query_pubmed() instead, or arXiv again later."
    response, why = _fetch_arxiv(params)
    if response is None:
        return _search_unavailable("arXiv", why, instead)
    entries = feedparser.parse(response.content).entries
    # arXiv explains a request it refuses, such as a query it cannot parse, in an
    # entry of its own: the fix is to the query, and waiting would not help.
    refusals = [entry.get("summary", "") for entry in entries if "arxiv.org/api/errors" in entry.get("id", "")]
    if refusals:
        return f"arXiv refused the query: {refusals[0].rstrip('.')}. Correct the query and search again."
    if response.status_code != requests.codes.ok:
        return _search_unavailable("arXiv", why, instead)
    papers = [
        f"Title: {' '.join(entry.get('title', '').split())}\nSummary: {entry.get('summary', '')}" for entry in entries
    ]
    return "\n\n".join(papers) if papers else "No papers found on arXiv."


def query_scholar(query: str) -> str:
    """Query Google Scholar for papers based on the provided search query.

    Parameters
    ----------
    - query (str): The search query string.

    Returns
    -------
    - str: The first search result formatted or an error message.

    """
    from scholarly import MaxTriesExceededException, scholarly

    # Straight to Google Scholar. Queries used to go through free public proxies
    # (scholarly's FreeProxies): servers nobody vetted saw every query, and since
    # httpx 0.28 their setup crashed before any search was made.
    instead = "Search PubMed with query_pubmed() or arXiv with query_arxiv() instead."
    try:
        result = next(scholarly.search_pubs(query), None)
    except MaxTriesExceededException:
        return _search_unavailable(
            "Google Scholar",
            "Google Scholar refused the requests, as it does once it notices automated searches",
            instead,
        )
    except Exception as e:
        return _search_unavailable("Google Scholar", f"the request failed ({type(e).__name__}: {e})", instead)
    if result is None:
        return "No results found on Google Scholar."
    bib = result.get("bib", {})
    return (
        f"Title: {bib.get('title', '')}\nYear: {bib.get('pub_year', '')}\n"
        f"Venue: {bib.get('venue', '')}\nAbstract: {bib.get('abstract', '')}"
    )


# NCBI's E-utilities take at most three requests a second from an IP, or ten
# with an API key (https://www.ncbi.nlm.nih.gov/books/NBK25497/), and answer a
# faster client with HTTP 429. Every query_pubmed call in the process - every
# chat's - takes its start time from this one schedule, where a client per call
# kept none. Requests may overlap: the limit is on how often they start.
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EUTILS_SPACING_S = 0.34
_EUTILS_SPACING_WITH_KEY_S = 0.11
_EUTILS_TIMEOUT_S = (10, 60)
_EUTILS_RETRY_WAITS_S = (1.0, 2.0, 4.0)
_EUTILS_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_eutils_lock = threading.Lock()
_eutils_next_start = 0.0  # time.monotonic() before which no request may start


def _eutils_request(utility: str, params: dict) -> requests.Response:
    """One E-utilities request, started once NCBI's rate limit allows.

    It is a POST: a GET would carry ``NCBI_API_KEY`` in its URL, which a
    connection error quotes, and that message reaches the agent and the log.
    """
    global _eutils_next_start
    api_key = credentials.getenv("NCBI_API_KEY")
    identity = {"tool": "Biomni-AD", "email": os.environ.get("NCBI_EMAIL"), "api_key": api_key}
    with _eutils_lock:
        now = time.monotonic()
        start = max(now, _eutils_next_start)
        _eutils_next_start = start + (_EUTILS_SPACING_WITH_KEY_S if api_key else _EUTILS_SPACING_S)
    time.sleep(start - now)
    data = {**params, **{name: value for name, value in identity.items() if value}}
    return requests.post(f"{_EUTILS}/{utility}", data=data, timeout=_EUTILS_TIMEOUT_S)


def _describe_eutils(response: requests.Response) -> str:
    why = f"NCBI answered HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        body = None
    # NCBI's own explanation, such as "API rate limit exceeded". Only that field:
    # the rest of an error body echoes the API key it was sent.
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, str) and error:
        why += f" ({error})"
    return why


def _eutils(utility: str, params: dict) -> tuple[requests.Response | None, str]:
    """NCBI's 200 answer to ``params``, or ``None`` and why there is none."""
    response, why, _ = _get_with_retries(
        lambda: _eutils_request(utility, params), _describe_eutils, _EUTILS_RETRY_WAITS_S, _EUTILS_RETRY_STATUSES
    )
    if response is not None and response.status_code != requests.codes.ok:
        return None, why
    return response, why


def _pubmed_search(term: str, max_papers: int) -> tuple[dict | None, str]:
    """PubMed's search result for ``term``, most relevant first, or ``None`` and why there is none."""
    params = {"db": "pubmed", "term": term, "retmax": max_papers, "sort": "relevance", "retmode": "json"}
    response, why = _eutils("esearch.fcgi", params)
    if response is None:
        return None, why
    try:
        result = response.json()["esearchresult"]
    except (ValueError, KeyError, TypeError):
        return None, "NCBI answered with something other than search results"
    if not isinstance(result, dict):
        return None, "NCBI answered with something other than search results"
    if result.get("ERROR"):
        return None, f"NCBI could not run the search ({result['ERROR']})"
    return result, ""


def _pubmed_records(pmids: list[str]) -> tuple[list[dict[str, str]] | None, str]:
    """The PubMed records of ``pmids``, in that order, or ``None`` and why there are none."""
    response, why = _eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    if response is None:
        return None, why
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return None, "NCBI answered with something other than PubMed records"
    records = [_pubmed_record(r) for r in root if r.tag in ("PubmedArticle", "PubmedBookArticle")]
    order = {pmid: i for i, pmid in enumerate(pmids)}
    return sorted(records, key=lambda r: order.get(r["pmid"], len(order))), ""


def _xml_text(element: ET.Element | None) -> str:
    """All of ``element``'s text, including what is inside markup such as ``<i>`` or ``<sub>``."""
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def _pubmed_record(record: ET.Element) -> dict[str, str]:
    """A PubmedArticle, or a PubmedBookArticle such as a GeneReviews chapter, as query_pubmed shows it."""
    if record.tag == "PubmedBookArticle":
        doc = "BookDocument"
        book_title = _xml_text(record.find(f"{doc}/Book/BookTitle"))
        publisher = _xml_text(record.find(f"{doc}/Book/Publisher/PublisherName"))
        title = _xml_text(record.find(f"{doc}/ArticleTitle")) or book_title
        venue = f"{book_title} ({publisher})" if publisher else book_title
        dates: tuple[str, ...] = (f"{doc}/ContributionDate/Year", f"{doc}/Book/PubDate/Year")
        abstract = record.find(f"{doc}/Abstract")
        pmid = record.findtext(f"{doc}/PMID", "")
    else:
        article = "MedlineCitation/Article"
        title = _xml_text(record.find(f"{article}/ArticleTitle"))
        venue = _xml_text(record.find(f"{article}/Journal/Title"))
        issue_date = f"{article}/Journal/JournalIssue/PubDate"
        dates = (f"{issue_date}/Year", f"{issue_date}/MedlineDate", f"{article}/ArticleDate/Year")
        abstract = record.find(f"{article}/Abstract")
        pmid = record.findtext("MedlineCitation/PMID", "")
    year = next((m.group() for path in dates if (m := re.search(r"\d{4}", record.findtext(path, "")))), "")
    sections = []
    parts = abstract.findall("AbstractText") if abstract is not None else []
    for part in parts:
        text = _xml_text(part)
        label = part.get("Label")
        if text:
            sections.append(f"{label}: {text}" if label and label != "UNLABELLED" else text)
    return {"pmid": pmid, "title": title, "abstract": "\n".join(sections), "venue": venue, "year": year}


_BOOLEAN_OPERATORS = frozenset({"AND", "OR", "NOT"})


def _shorter_queries(query: str, max_shorter: int) -> list[str]:
    """``query``, then up to ``max_shorter`` distinct versions of it with the last words dropped."""
    words = query.split()
    queries = [query]
    for drop in range(1, max_shorter + 1):
        kept = words[:-drop]
        while kept and kept[-1].upper() in _BOOLEAN_OPERATORS:
            kept.pop()
        shorter = " ".join(kept)
        if shorter and shorter not in queries:
            queries.append(shorter)
    return queries


def _quoted(items: list[str], conjunction: str) -> str:
    quoted = [f'"{item}"' for item in items]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} {conjunction} {quoted[-1]}"


def query_pubmed(query: str, max_papers: int = 10, max_retries: int = 3) -> str:
    """Query PubMed for papers based on the provided search query, most relevant first.

    When nothing matches, the query is tried again, up to ``max_retries`` times,
    with its last words dropped; the answer then says which query found the papers.

    Parameters
    ----------
    - query (str): The search query string.
    - max_papers (int): The maximum number of papers to retrieve (default: 10).
    - max_retries (int): Maximum number of retries with a shorter query (default: 3).

    Returns
    -------
    - str: Title, abstract, journal, year and PMID of each paper, or why there are none.

    """
    instead = "Search arXiv with query_arxiv() or Google Scholar with query_scholar() instead, or PubMed again later."
    tried: list[str] = []
    for term in _shorter_queries(query, max_retries):
        found, why = _pubmed_search(term, max_papers)
        if found is None:
            return _search_unavailable("PubMed", why, instead)
        tried.append(term)
        if found.get("idlist"):
            break
    else:
        return f"No papers found on PubMed for {_quoted(tried, 'or')}."
    papers, why = _pubmed_records(found["idlist"])
    if papers is None:
        return _search_unavailable("PubMed", why, instead)
    if not papers:
        return _search_unavailable("PubMed", "NCBI listed matching papers but sent none of their records", instead)
    notes = []
    if term != query:
        notes.append(f'No papers matched "{query}" on PubMed, so these are for the shorter query "{term}".')
    # PubMed leaves out of a search the words it does not know, and says so only here.
    unknown = (found.get("errorlist") or {}).get("phrasesnotfound") or []
    if unknown:
        notes.append(
            f"PubMed does not recognise {_quoted(unknown, 'and')}, so it searched without "
            f"{'it' if len(unknown) == 1 else 'them'}."
        )
    results = [
        f"Title: {p['title']}\nAbstract: {p['abstract']}\nJournal: {p['venue']}\nYear: {p['year']}\nPMID: {p['pmid']}"
        for p in papers
    ]
    return "\n\n".join([" ".join(notes), *results] if notes else results)


def search_google(query: str, num_results: int = 3, language: str = "en") -> str:
    """Search using Google search.

    Args:
        query (str): The search query (e.g., "protocol text or search question")
        num_results (int): Number of results to return (default: 3)
        language (str): Language code for search results (default: 'en')

    Returns:
        str: One Title / URL / Description block per result, or a sentence
        saying that Google could not be searched

    """
    instead = "Search the literature with query_pubmed(), query_arxiv() or query_scholar() instead."
    results_string = ""
    print(f"Searching for {query} with {num_results} results and {language} language")
    try:
        for res in search(query, num_results=num_results, lang=language, advanced=True):
            print(f"Found result: {res.title}")
            results_string += f"Title: {res.title}\nURL: {res.url}\nDescription: {res.description}\n\n"
    except Exception as e:
        return _search_unavailable("Google", f"the request failed ({type(e).__name__}: {e})", instead)
    if not results_string:
        # Google answers automated requests with an ordinary HTTP 200 page that
        # holds no results ("Your browser isn't supported anymore", or a
        # JavaScript challenge), so every query comes back empty without an
        # error. Returned as it was, that emptiness read as "nothing on the web".
        return _search_unavailable(
            "Google", "Google returned a page without results, as it does for automated searches", instead
        )
    return results_string


def advanced_web_search_claude(
    query: str,
    max_searches: int = 1,
    max_retries: int = 3,
) -> str:
    """
    Initiate an advanced web search by launching a specialized agent to collect relevant information and citations through multiple rounds of web searches for a given query.
    Craft the query carefully for the search agent to find the most relevant information.

    Parameters
    ----------
    query : str
        The search phrase you want Claude to look up.
    max_searches : int, optional
        Upper-bound on searches Claude may issue inside this request.
    max_retries : int, optional
        Maximum number of retry attempts with exponential backoff.

    Returns
    -------
    full_text : str
        A formatted string containing the full text response from Claude and the citations.
    """
    import random

    import anthropic

    from biomni.config import default_config
    from biomni.tool.availability import claude_web_search_problem

    # Checked here as well as when the tool list is built, because generated
    # code can import the function whether or not it was advertised - and
    # behind the LLM proxy a direct call to Anthropic must never be made.
    problem = claude_web_search_problem()
    if problem:
        raise RuntimeError(problem)

    # ANTHROPIC_API_KEY only. default_config.api_key belongs to a custom model
    # endpoint, and sending it to api.anthropic.com would hand that endpoint's
    # credential to a third party.
    client = anthropic.Anthropic(api_key=credentials.getenv("ANTHROPIC_API_KEY"))
    model = default_config.llm
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_searches,
    }

    delay = random.randint(1, 10)
    error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": query}],
                tools=[tool_def],
            )

            paragraphs, citations = [], []
            response.content = response.content
            formatted_response = ""
            for blk in response.content:
                if blk.type == "text":
                    paragraphs.append(blk.text)
                    formatted_response += blk.text

                    if blk.citations:
                        for cite in blk.citations:
                            citations.append({"url": cite.url, "title": cite.title, "cited_text": cite.cited_text})
                            formatted_response += f"(Citation: {cite.title} - {cite.url})"
            return formatted_response

        except Exception as e:
            error = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    print(f"Error performing web search after {max_retries} attempts: {error}")
    return f"Error performing web search after {max_retries} attempts: {error}"


def extract_url_content(url: str) -> str:
    """Extract the text content of a webpage using requests and BeautifulSoup.

    Args:
        url: Webpage URL to extract content from

    Returns:
        Text content of the webpage

    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)

    # Check if the response is in text format
    if "text/plain" in response.headers.get("Content-Type", "") or "application/json" in response.headers.get(
        "Content-Type", ""
    ):
        return response.text.strip()  # Return plain text or JSON response directly

    # If it's HTML, use BeautifulSoup to parse
    soup = BeautifulSoup(response.text, "html.parser")

    # Try to find main content first, fallback to body
    content = soup.find("main") or soup.find("article") or soup.body

    # Remove unwanted elements
    for element in content(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        element.decompose()

    # Extract text with better formatting
    paragraphs = content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_text = []

    for p in paragraphs:
        text = p.get_text().strip()
        if text:  # Only add non-empty paragraphs
            cleaned_text.append(text)

    return "\n\n".join(cleaned_text)


def extract_pdf_content(url: str) -> str:
    """Extract the text content of a PDF file given its URL.

    Args:
        url: URL of the PDF file to extract text from

    Returns:
        The extracted text content from the PDF

    """
    try:
        # Check if the URL ends with .pdf
        if not url.lower().endswith(".pdf"):
            # If not, try to find a PDF link on the page
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Look for PDF links in the HTML content
                pdf_links = re.findall(r'href=[\'"]([^\'"]+\.pdf)[\'"]', response.text)
                if pdf_links:
                    # Use the first PDF link found
                    if not pdf_links[0].startswith("http"):
                        # Handle relative URLs
                        base_url = "/".join(url.split("/")[:3])
                        url = base_url + pdf_links[0] if pdf_links[0].startswith("/") else base_url + "/" + pdf_links[0]
                    else:
                        url = pdf_links[0]
                else:
                    return f"No PDF file found at {url}. Please provide a direct link to a PDF file."

        # Download the PDF
        response = requests.get(url, timeout=30)

        # Check if we actually got a PDF file (by checking content type or magic bytes)
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type and not response.content.startswith(b"%PDF"):
            return f"The URL did not return a valid PDF file. Content type: {content_type}"

        pdf_file = BytesIO(response.content)

        # Try with PyPDF2 first
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")

        # Clean up the text
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return "The PDF file did not contain any extractable text. It may be an image-based PDF requiring OCR."

        return text

    except requests.exceptions.RequestException as e:
        return f"Error downloading PDF: {str(e)}"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
