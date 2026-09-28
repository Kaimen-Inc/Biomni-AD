"""Tests for the literature search tools: a search that could not be made says so.

An empty string, or "no papers found", reads to the agent as "nothing exists on
this", and it carried on as if it had searched. Google now answers automated
requests with a page that holds no results, arXiv answers the clients it
throttles with HTTP 406, Google Scholar blocks automated searches it notices,
and NCBI answers a client over its rate limit with HTTP 429, so each of those
has to come back as a sentence that says the search did not happen.

PubMed results have to be whole, too: titles and abstracts used to be cut at
their first inline markup, and one book chapter among the results failed the
whole search.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from types import SimpleNamespace

import pytest
import requests
from biomni import credentials

literature = pytest.importorskip("biomni.tool.literature")

# A real export.arxiv.org response, trimmed to two entries with short summaries.
# The first title keeps the line break arXiv puts in long titles.
ARXIV_FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns="http://www.w3.org/2005/Atom">
  <opensearch:totalResults>379</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2010.07473v1</id>
    <title>Modeling Microglia Activation and Inflammation-Based
  Neuroprotectant Strategies During Ischemic Stroke</title>
    <summary>Neural inflammation immediately follows the onset of ischemic stroke.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1802.04156v1</id>
    <title>The Coupled TuFF-BFF Algorithm for Automatic 3D Segmentation of Microglia</title>
    <summary>We propose an automatic 3D segmentation algorithm for images of microglia.</summary>
  </entry>
</feed>
"""
EMPTY_FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>arXiv Query</title></feed>
"""
# export.arxiv.org's answer to "TREM2 AND (microglia", as it sent it, with HTTP 400.
ARXIV_REFUSAL = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/</id>
  <title>arXiv Search Results</title>
  <updated>2026-09-28T12:22:04Z</updated>
  <opensearch:itemsPerPage>1</opensearch:itemsPerPage>
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>https://arxiv.org/api/errors</id>
    <title>Error</title>
    <updated>2026-09-28T12:22:04Z</updated>
    <link href="https://arxiv.org/api/errors" rel="alternate" type="text/html"/>
    <summary>Invalid query string: 'all:TREM2 AND (all:microglia'</summary>
    <author>
      <name>arXiv api core</name>
    </author>
  </entry>
</feed>
"""


class FakeClock:
    """Stands in for the time module: sleeping moves the clock, instantly."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        if seconds > 0:
            self.sleeps.append(seconds)
            self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(literature, "time", fake)
    monkeypatch.setattr(literature, "_arxiv_last_request", 0.0)
    monkeypatch.setattr(literature, "_arxiv_resting_until", 0.0)
    return fake


@pytest.fixture
def arxiv(monkeypatch, clock):
    """A scripted export.arxiv.org: each request gets the next answer in ``answers``."""
    server = SimpleNamespace(answers=[], requests=[])

    def fake_get(url, params=None, headers=None, timeout=None):
        server.requests.append({"url": url, "params": params, "headers": headers, "timeout": timeout, "at": clock.now})
        answer = server.answers.pop(0) if len(server.answers) > 1 else server.answers[0]
        if isinstance(answer, Exception):
            raise answer
        status, content = answer
        return SimpleNamespace(status_code=status, content=content)

    monkeypatch.setattr(literature.requests, "get", fake_get)
    return server


def test_arxiv_results_are_titles_and_summaries(arxiv):
    arxiv.answers = [(200, ARXIV_FEED)]

    out = literature.query_arxiv("TREM2 microglia", max_papers=2)

    assert out == (
        "Title: Modeling Microglia Activation and Inflammation-Based Neuroprotectant Strategies During Ischemic Stroke\n"
        "Summary: Neural inflammation immediately follows the onset of ischemic stroke.\n\n"
        "Title: The Coupled TuFF-BFF Algorithm for Automatic 3D Segmentation of Microglia\n"
        "Summary: We propose an automatic 3D segmentation algorithm for images of microglia."
    )
    (sent,) = arxiv.requests
    assert sent["url"] == "https://export.arxiv.org/api/query"
    assert sent["params"]["search_query"] == "TREM2 microglia"
    assert sent["params"]["max_results"] == 2
    assert sent["headers"]["Accept"] == "application/atom+xml"
    assert sent["timeout"], "a request without a timeout can hold the code step forever"


def test_arxiv_with_no_matches_says_so(arxiv):
    arxiv.answers = [(200, EMPTY_FEED)]

    assert literature.query_arxiv("nothing matches this") == "No papers found on arXiv."


def test_a_throttled_arxiv_request_is_retried_after_growing_waits(arxiv, clock):
    arxiv.answers = [(406, b""), (406, b""), (200, ARXIV_FEED)]

    out = literature.query_arxiv("TREM2 microglia", max_papers=2)

    assert out.startswith("Title: Modeling Microglia Activation")
    assert len(arxiv.requests) == 3
    assert [s for s in clock.sleeps if s in literature._ARXIV_RETRY_WAITS_S] == [5.0, 10.0]


def test_an_arxiv_that_keeps_throttling_is_reported_not_returned_empty(arxiv, caplog):
    arxiv.answers = [(406, b"")]

    with caplog.at_level(logging.WARNING, logger="biomni.tool.literature"):
        out = literature.query_arxiv("TREM2 microglia")

    assert out.startswith("arXiv search is unavailable: arXiv answered HTTP 406")
    assert "throttling" in out
    assert "on the last of 4 tries over 35 seconds" in out
    assert "This is not an empty result" in out
    assert "query_pubmed()" in out
    assert len(arxiv.requests) == 4
    assert "arXiv search unavailable" in caplog.text


def test_arxiv_is_left_alone_for_a_while_once_it_keeps_throttling(arxiv, clock):
    arxiv.answers = [(406, b"")]
    literature.query_arxiv("TREM2 microglia")
    assert len(arxiv.requests) == 4

    # Straight after, no request at all: the same answer, without the waits.
    out = literature.query_arxiv("APOE4 astrocytes")
    assert "not asked again" in out
    assert out.startswith("arXiv search is unavailable")
    assert len(arxiv.requests) == 4

    # Once the rest is over, arXiv is asked again.
    clock.now += literature._ARXIV_REST_S
    arxiv.answers = [(200, ARXIV_FEED)]
    assert literature.query_arxiv("APOE4 astrocytes").startswith("Title: ")
    assert len(arxiv.requests) == 5


def test_an_arxiv_refusal_that_is_not_throttling_is_not_retried(arxiv, clock):
    arxiv.answers = [(400, b"")]

    out = literature.query_arxiv("ti:")

    assert out.startswith("arXiv search is unavailable: arXiv answered HTTP 400.")
    assert "throttling" not in out
    assert len(arxiv.requests) == 1
    # Not throttled, so the next call asks arXiv again straight away.
    literature.query_arxiv("ti:")
    assert len(arxiv.requests) == 2


def test_arxiv_explains_a_query_it_refused(arxiv, caplog):
    # The fix is to the query, so the agent reads arXiv's reason, not "try later".
    arxiv.answers = [(400, ARXIV_REFUSAL)]

    with caplog.at_level(logging.WARNING, logger="biomni.tool.literature"):
        out = literature.query_arxiv("TREM2 AND (microglia")

    assert out == (
        "arXiv refused the query: Invalid query string: 'all:TREM2 AND (all:microglia'. "
        "Correct the query and search again."
    )
    assert len(arxiv.requests) == 1
    assert "search unavailable" not in caplog.text, "a malformed query is not the operator's to fix"


def test_arxiv_connection_errors_are_retried(arxiv):
    arxiv.answers = [requests.ConnectionError("reset by peer"), (200, ARXIV_FEED)]

    assert literature.query_arxiv("TREM2 microglia").startswith("Title: ")
    assert len(arxiv.requests) == 2


def test_an_arxiv_that_cannot_be_reached_is_reported_and_left_alone(arxiv):
    # What a deployment whose egress does not allow export.arxiv.org sees.
    arxiv.answers = [requests.ConnectionError("Failed to resolve 'export.arxiv.org'")]

    out = literature.query_arxiv("TREM2 microglia")

    assert out.startswith(
        "arXiv search is unavailable: the request failed (ConnectionError: Failed to resolve 'export.arxiv.org'), "
        "on the last of 4 tries over 35 seconds."
    )
    assert len(arxiv.requests) == 4
    # The next call answers at once, instead of waiting out the retries again.
    assert "not asked again" in literature.query_arxiv("APOE4 astrocytes")
    assert len(arxiv.requests) == 4


def test_arxiv_requests_are_spaced_across_calls(arxiv, clock):
    # arXiv asks for at most one request every three seconds - across every
    # chat in the process, not only within one call.
    arxiv.answers = [(200, ARXIV_FEED)]

    literature.query_arxiv("TREM2 microglia")
    literature.query_arxiv("APOE4 astrocytes")

    first, second = arxiv.requests
    assert second["at"] - first["at"] >= literature._ARXIV_SPACING_S


def _google_results(*results):
    def fake_search(term, num_results, lang, advanced):
        assert advanced
        yield from (SimpleNamespace(url=u, title=t, description=d) for u, t, d in results)

    return fake_search


def test_google_results_are_title_url_and_description(monkeypatch):
    monkeypatch.setattr(
        literature,
        "search",
        _google_results(("https://example.org/trem2", "TREM2 in microglia", "A review of TREM2 signalling.")),
    )

    out = literature.search_google("TREM2 microglia")

    assert (
        out
        == "Title: TREM2 in microglia\nURL: https://example.org/trem2\nDescription: A review of TREM2 signalling.\n\n"
    )


def test_google_answering_without_results_is_reported_as_unavailable(monkeypatch, caplog):
    # What Google sends an automated request today: HTTP 200, and a page with
    # no results in it, so the scraper yields nothing and raises nothing.
    monkeypatch.setattr(literature, "search", _google_results())

    with caplog.at_level(logging.WARNING, logger="biomni.tool.literature"):
        out = literature.search_google("TREM2 microglia")

    assert out.startswith("Google search is unavailable: Google returned a page without results")
    assert "This is not an empty result" in out
    assert "query_pubmed()" in out
    assert "Google search unavailable" in caplog.text


def test_a_failed_google_request_is_reported_as_unavailable(monkeypatch):
    def refused(term, num_results, lang, advanced):
        raise requests.HTTPError("429 Client Error: Too Many Requests for url: https://www.google.com/search")
        yield  # pragma: no cover - makes this a generator, like the real one

    monkeypatch.setattr(literature, "search", refused)

    out = literature.search_google("TREM2 microglia")

    assert out.startswith("Google search is unavailable: the request failed (HTTPError: 429 Client Error")


class FakeScholar:
    """The slice of scholarly's module-level ``scholarly`` object the tool uses."""

    def __init__(self, results=(), error=None):
        self.results = list(results)
        self.error = error
        self.queries: list[str] = []

    def search_pubs(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return iter(self.results)


class MaxTriesExceededException(Exception):
    pass


@pytest.fixture
def scholar(monkeypatch):
    """Installs a stand-in scholarly package; tests set ``.results`` or ``.error``.

    It has no ProxyGenerator and no use_proxy: routing queries through proxies
    would fail here, as it should.
    """
    fake = FakeScholar()
    module = types.ModuleType("scholarly")
    module.scholarly = fake
    module.MaxTriesExceededException = MaxTriesExceededException
    monkeypatch.setitem(sys.modules, "scholarly", module)
    return fake


def test_scholar_returns_the_first_result_without_any_proxy(scholar):
    scholar.results = [
        {
            "bib": {
                "title": "TREM2-mediated early microglial response limits diffusion and toxicity of amyloid plaques",
                "pub_year": "2016",
                "venue": "Journal of Experimental Medicine",
                "abstract": "Variants of TREM2 are associated with Alzheimer's disease.",
            }
        },
        {"bib": {"title": "A second result"}},
    ]

    out = literature.query_scholar("TREM2 microglia")

    assert out == (
        "Title: TREM2-mediated early microglial response limits diffusion and toxicity of amyloid plaques\n"
        "Year: 2016\nVenue: Journal of Experimental Medicine\n"
        "Abstract: Variants of TREM2 are associated with Alzheimer's disease."
    )
    assert scholar.queries == ["TREM2 microglia"]


def test_a_scholar_result_missing_fields_still_reads(scholar):
    scholar.results = [{"bib": {"title": "TREM2, microglia, and neurodegenerative diseases"}}]

    out = literature.query_scholar("TREM2 microglia")

    assert out == "Title: TREM2, microglia, and neurodegenerative diseases\nYear: \nVenue: \nAbstract: "


def test_scholar_with_no_matches_says_so(scholar):
    assert literature.query_scholar("nothing matches this") == "No results found on Google Scholar."


def test_a_blocked_scholar_is_reported_as_unavailable(scholar):
    scholar.error = MaxTriesExceededException("Cannot Fetch from Google Scholar.")

    out = literature.query_scholar("TREM2 microglia")

    assert out.startswith("Google Scholar search is unavailable: Google Scholar refused the requests")
    assert "This is not an empty result" in out
    assert "query_pubmed()" in out


def test_a_failed_scholar_request_is_reported_as_unavailable(scholar):
    scholar.error = TypeError("Client.__init__() got an unexpected keyword argument 'proxies'")

    out = literature.query_scholar("TREM2 microglia")

    assert out.startswith("Google Scholar search is unavailable: the request failed (TypeError: Client.__init__()")


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.content = body if isinstance(body, bytes) else json.dumps(body).encode()

    def json(self):
        return json.loads(self.content)


def esearch(*pmids, **fields):
    """An esearch.fcgi JSON answer listing ``pmids``."""
    result = {"count": str(len(pmids)), "retmax": str(len(pmids)), "retstart": "0", "idlist": list(pmids), **fields}
    return {"header": {"type": "esearch", "version": "0.3"}, "esearchresult": result}


# The body NCBI sends with HTTP 429. Without an API key, "api-key" is the client's IP.
RATE_LIMITED = {"error": "API rate limit exceeded", "api-key": "192.0.2.1", "count": "4", "limit": "3"}

# A real efetch.fcgi answer for 40346446, 32613609 and 20301340, trimmed to the
# elements the tool reads and to shorter text. The records keep their inline
# markup (<sub>, <sup>), labelled sections and the book chapter's layout, and
# are not in the order the search listed them.
PUBMED_RECORDS = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2025//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_250101.dtd">
<PubmedArticleSet>
<PubmedArticle><MedlineCitation Status="MEDLINE" Owner="NLM"><PMID Version="1">40346446</PMID><Article PubModel="Print">
<Journal><JournalIssue CitedMedium="Internet"><Volume>21</Volume><Issue>5</Issue><PubDate><Year>2025</Year><Month>May</Month></PubDate></JournalIssue>
<Title>Alzheimer's &amp; dementia : the journal of the Alzheimer's Association</Title><ISOAbbreviation>Alzheimers Dement</ISOAbbreviation></Journal>
<ArticleTitle>PLCG2 modulates TREM2 expression and signaling in response to Alzheimer's disease pathology.</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Phospholipase C gamma 2 (PLCG2) is an intracellular effector of microglial cell surface receptors.</AbstractText>
<AbstractText Label="METHODS">5xFAD mice were crossed with PLCG2- and TREM2-deficient mice.</AbstractText>
<CopyrightInformation>© 2025 The Author(s).</CopyrightInformation></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation Status="MEDLINE" Owner="NLM"><PMID Version="1">32613609</PMID><Article PubModel="Print-Electronic">
<Journal><JournalIssue CitedMedium="Internet"><Volume>34</Volume><Issue>8</Issue><PubDate><Year>2020</Year><Month>Aug</Month></PubDate></JournalIssue>
<Title>FASEB journal : official publication of the Federation of American Societies for Experimental Biology</Title></Journal>
<ArticleTitle>Activation of FAK/Rac1/Cdc42-GTPase signaling ameliorates impaired microglial migration response to Aβ<sub>42</sub> in triggering receptor expressed on myeloid cells 2 loss-of-function murine models.</ArticleTitle>
<Abstract><AbstractText>Here, we confirm that TREM2 mutation attenuates microglial migration. Then, using Trem2<sup>-/-</sup> mice and an R47H variant mouse model for AD generated for this study, we show that TREM2 deficiency inhibits FAK and Rac1/Cdc42-GTPase signaling.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedBookArticle><BookDocument><PMID Version="1">20301340</PMID>
<Book><Publisher><PublisherName>University of Washington, Seattle</PublisherName><PublisherLocation>Seattle (WA)</PublisherLocation></Publisher><BookTitle book="gene">GeneReviews<sup>®</sup></BookTitle><PubDate><Year>1993</Year></PubDate></Book>
<ArticleTitle book="gene" part="alzheimer">Alzheimer Disease Overview</ArticleTitle>
<Abstract><AbstractText>The purpose of this overview is to: 1.. Describe the clinical characteristics of Alzheimer disease (AD).</AbstractText></Abstract>
<ContributionDate><Year>1998</Year><Month>10</Month><Day>23</Day></ContributionDate>
</BookDocument></PubmedBookArticle>
</PubmedArticleSet>
""".encode()


@pytest.fixture
def ncbi(monkeypatch, clock):
    """A scripted eutils.ncbi.nlm.nih.gov: each request to a utility gets its next answer."""
    server = SimpleNamespace(answers={"esearch.fcgi": [], "efetch.fcgi": []}, requests=[])

    def fake_post(url, data=None, timeout=None, **kwargs):
        utility = url.rsplit("/", 1)[-1]
        server.requests.append({"url": url, "utility": utility, "data": data, "timeout": timeout, "at": clock.now})
        queue = server.answers[utility]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(*answer)

    monkeypatch.setattr(literature.requests, "post", fake_post)
    monkeypatch.setattr(literature, "_eutils_next_start", 0.0)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    return server


def test_pubmed_results_are_whole_and_most_relevant_first(ncbi):
    ncbi.answers["esearch.fcgi"] = [(200, esearch("32613609", "20301340", "40346446"))]
    ncbi.answers["efetch.fcgi"] = [(200, PUBMED_RECORDS)]

    out = literature.query_pubmed("TREM2 microglia", max_papers=3)

    assert out == (
        "Title: Activation of FAK/Rac1/Cdc42-GTPase signaling ameliorates impaired microglial migration response to "
        "Aβ42 in triggering receptor expressed on myeloid cells 2 loss-of-function murine models.\n"
        "Abstract: Here, we confirm that TREM2 mutation attenuates microglial migration. Then, using Trem2-/- mice and "
        "an R47H variant mouse model for AD generated for this study, we show that TREM2 deficiency inhibits FAK and "
        "Rac1/Cdc42-GTPase signaling.\n"
        "Journal: FASEB journal : official publication of the Federation of American Societies for Experimental Biology\n"
        "Year: 2020\nPMID: 32613609\n\n"
        "Title: Alzheimer Disease Overview\n"
        "Abstract: The purpose of this overview is to: 1.. Describe the clinical characteristics of Alzheimer disease (AD).\n"
        "Journal: GeneReviews® (University of Washington, Seattle)\nYear: 1998\nPMID: 20301340\n\n"
        "Title: PLCG2 modulates TREM2 expression and signaling in response to Alzheimer's disease pathology.\n"
        "Abstract: BACKGROUND: Phospholipase C gamma 2 (PLCG2) is an intracellular effector of microglial cell surface "
        "receptors.\nMETHODS: 5xFAD mice were crossed with PLCG2- and TREM2-deficient mice.\n"
        "Journal: Alzheimer's & dementia : the journal of the Alzheimer's Association\nYear: 2025\nPMID: 40346446"
    )
    search, fetch = ncbi.requests
    assert search["utility"] == "esearch.fcgi"
    assert search["data"] == {
        "db": "pubmed",
        "term": "TREM2 microglia",
        "retmax": 3,
        "sort": "relevance",
        "retmode": "json",
        "tool": "Biomni-AD",
    }
    assert fetch["data"]["id"] == "32613609,20301340,40346446"
    assert search["timeout"] and fetch["timeout"], "a request without a timeout can hold the code step forever"


def test_pubmed_says_which_words_it_searched_without(ncbi):
    # PubMed drops words it does not know and searches the rest: "TREM2 xyzzyqqq"
    # is a search for TREM2.
    unknown = {"phrasesnotfound": ["xyzzyqqq"], "fieldsnotfound": []}
    ncbi.answers["esearch.fcgi"] = [(200, esearch("32613609", errorlist=unknown))]
    ncbi.answers["efetch.fcgi"] = [(200, PUBMED_RECORDS)]

    out = literature.query_pubmed("TREM2 xyzzyqqq", max_papers=1)

    assert out.startswith('PubMed does not recognise "xyzzyqqq", so it searched without it.\n\nTitle: Activation')


def test_pubmed_says_when_its_results_are_for_a_shorter_query(ncbi):
    ncbi.answers["esearch.fcgi"] = [(200, esearch()), (200, esearch("32613609"))]
    ncbi.answers["efetch.fcgi"] = [(200, PUBMED_RECORDS)]

    out = literature.query_pubmed("TREM2 microglia AND zebrafish", max_papers=1)

    assert out.startswith(
        'No papers matched "TREM2 microglia AND zebrafish" on PubMed, so these are for the shorter query '
        '"TREM2 microglia".\n\nTitle: Activation'
    )
    # The dangling AND goes with the word after it.
    assert [r["data"]["term"] for r in ncbi.requests if r["utility"] == "esearch.fcgi"] == [
        "TREM2 microglia AND zebrafish",
        "TREM2 microglia",
    ]


def test_pubmed_with_no_matches_says_what_it_tried(ncbi):
    ncbi.answers["esearch.fcgi"] = [(200, esearch())]

    out = literature.query_pubmed("xyzzyqqq plorkfn")

    assert out == 'No papers found on PubMed for "xyzzyqqq plorkfn" or "xyzzyqqq".'
    assert len(ncbi.requests) == 2, "the one-word query is not sent three times over"


def test_pubmed_over_ncbis_rate_limit_is_retried(ncbi, clock):
    ncbi.answers["esearch.fcgi"] = [(429, RATE_LIMITED), (429, RATE_LIMITED), (200, esearch("32613609"))]
    ncbi.answers["efetch.fcgi"] = [(200, PUBMED_RECORDS)]

    out = literature.query_pubmed("TREM2 microglia", max_papers=1)

    assert out.startswith("Title: Activation")
    assert [s for s in clock.sleeps if s in literature._EUTILS_RETRY_WAITS_S] == [1.0, 2.0]


def test_pubmed_that_stays_over_the_rate_limit_is_reported_not_returned_empty(ncbi, caplog):
    ncbi.answers["esearch.fcgi"] = [(429, RATE_LIMITED)]

    with caplog.at_level(logging.WARNING, logger="biomni.tool.literature"):
        out = literature.query_pubmed("TREM2 microglia")

    assert out.startswith(
        "PubMed search is unavailable: NCBI answered HTTP 429 (API rate limit exceeded), "
        "on the last of 4 tries over 7 seconds."
    )
    assert "This is not an empty result" in out
    assert "query_arxiv()" in out
    assert "192.0.2.1" not in out, "only NCBI's explanation is quoted, not the rest of its answer"
    assert "PubMed search unavailable" in caplog.text


def test_pubmed_requests_keep_to_ncbis_rate_across_calls(ncbi, monkeypatch):
    # Three requests a second from an IP - across every chat in the process.
    ncbi.answers["esearch.fcgi"] = [(200, esearch("32613609"))]
    ncbi.answers["efetch.fcgi"] = [(200, PUBMED_RECORDS)]

    literature.query_pubmed("TREM2 microglia", max_papers=1)
    literature.query_pubmed("APOE4 astrocytes", max_papers=1)
    starts = [r["at"] for r in ncbi.requests]
    assert len(starts) == 4
    assert all(b - a >= literature._EUTILS_SPACING_S - 1e-9 for a, b in zip(starts, starts[1:], strict=False))

    # Ten a second with an API key.
    monkeypatch.setenv("NCBI_API_KEY", "an-ncbi-key")
    ncbi.requests.clear()
    literature.query_pubmed("TREM2 microglia", max_papers=1)
    first, second = (r["at"] for r in ncbi.requests)
    assert literature._EUTILS_SPACING_WITH_KEY_S - 1e-9 <= second - first < literature._EUTILS_SPACING_S


def test_the_ncbi_api_key_is_sent_in_the_body_even_while_generated_code_runs(ncbi, monkeypatch):
    # Generated code runs with credentials scrubbed from os.environ, and it is
    # what calls query_pubmed. A key in the URL would be quoted by any
    # connection error, straight into the agent's transcript.
    monkeypatch.setenv("NCBI_API_KEY", "an-ncbi-key")
    monkeypatch.setenv("NCBI_EMAIL", "ops@example.org")
    ncbi.answers["esearch.fcgi"] = [(200, esearch("32613609"))]
    ncbi.answers["efetch.fcgi"] = [(200, PUBMED_RECORDS)]

    with credentials.scrubbed_environ():
        assert "NCBI_API_KEY" not in literature.os.environ
        literature.query_pubmed("TREM2 microglia", max_papers=1)

    assert len(ncbi.requests) == 2
    for sent in ncbi.requests:
        assert sent["data"]["api_key"] == "an-ncbi-key"
        assert sent["data"]["email"] == "ops@example.org"
        assert "?" not in sent["url"]


def test_an_ncbi_search_error_is_reported(ncbi):
    # NCBI's answer to an empty term: HTTP 200, and an error in place of results.
    ncbi.answers["esearch.fcgi"] = [(200, {"esearchresult": {"ERROR": "Empty term and query_key - nothing todo"}})]

    out = literature.query_pubmed("")

    assert out.startswith(
        "PubMed search is unavailable: NCBI could not run the search (Empty term and query_key - nothing todo)."
    )


def test_a_rejected_ncbi_api_key_is_reported_without_echoing_it(ncbi, monkeypatch):
    monkeypatch.setenv("NCBI_API_KEY", "not-a-real-key")
    # What NCBI answers, with HTTP 400: the body repeats the key.
    rejected = {"error": "API key invalid", "api-key": "not-a-real-key", "type": "invalid", "status": "unknown"}
    ncbi.answers["esearch.fcgi"] = [(400, rejected)]

    out = literature.query_pubmed("TREM2 microglia")

    assert out.startswith("PubMed search is unavailable: NCBI answered HTTP 400 (API key invalid).")
    assert "not-a-real-key" not in out
    assert len(ncbi.requests) == 1


def test_pubmed_records_ncbi_could_not_send_are_reported(ncbi):
    ncbi.answers["esearch.fcgi"] = [(200, esearch("32613609"))]
    ncbi.answers["efetch.fcgi"] = [(200, b"<html><body>Service unavailable</body>")]

    out = literature.query_pubmed("TREM2 microglia")

    assert out.startswith("PubMed search is unavailable: NCBI answered with something other than PubMed records.")


def test_pubmed_record_details(ncbi):
    # Shapes the trimmed records above do not have: a section NLM left
    # UNLABELLED, a translated abstract, and a season in place of a year.
    ncbi.answers["esearch.fcgi"] = [(200, esearch("31000001"))]
    ncbi.answers["efetch.fcgi"] = [
        (
            200,
            b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>31000001</PMID><Article>
<Journal><JournalIssue><PubDate><MedlineDate>2019 Nov-Dec</MedlineDate></PubDate></JournalIssue><Title>A journal</Title></Journal>
<ArticleTitle>A title</ArticleTitle>
<Abstract><AbstractText Label="UNLABELLED">The only section.</AbstractText></Abstract>
</Article><OtherAbstract Type="Publisher" Language="fre"><AbstractText>Un resume.</AbstractText></OtherAbstract>
</MedlineCitation></PubmedArticle></PubmedArticleSet>""",
        )
    ]

    out = literature.query_pubmed("a title")

    assert out == "Title: A title\nAbstract: The only section.\nJournal: A journal\nYear: 2019\nPMID: 31000001"
