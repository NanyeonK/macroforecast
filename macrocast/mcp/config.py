"""Paths, paper metadata, and shared configuration for the Coulombe RAG system."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Persistent storage paths
# ---------------------------------------------------------------------------

CACHE_ROOT = Path.home() / ".macrocast" / "rag"
BLOG_CACHE_DIR = CACHE_ROOT / "blog"
PAPERS_CACHE_DIR = CACHE_ROOT / "papers"
CHROMA_DIR = CACHE_ROOT / "chromadb"

# Source paths
PDF_SOURCE_DIR = Path.home() / "second_brain" / "research_kb" / "pdfs"
METHODOLOGY_DOC = (
    Path(__file__).parent.parent.parent
    / "docs"
    / "research"
    / "coulombe-methodology.md"
)

CHROMA_COLLECTION = "coulombe_knowledge"

# Embedding model (local inference via sentence-transformers, ~550 MB first download)
# Requires trust_remote_code=True — use make_embedding_function() helper below.
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def make_embedding_function():  # type: ignore[return]  # chromadb not a hard dependency
    """Return a ChromaDB-compatible embedding function for EMBED_MODEL.

    Uses sentence-transformers with trust_remote_code=True, which is required
    by the nomic-embed-text-v1.5 model.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)

    class _NomicEF:
        def name(self) -> str:  # required by ChromaDB
            return "nomic-embed-text-v1.5"

        def _encode(self, texts: list[str]) -> list[list[float]]:
            return model.encode(texts, convert_to_numpy=True).tolist()

        def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
            return self._encode(input)

        # ChromaDB ≥0.6 calls embed_query for query paths
        def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
            return self._encode(input)

    return _NomicEF()


# Blog API endpoint (Squarespace JSON feed)
BLOG_API_URL = "https://philippegouletcoulombe.com/blog?format=json"

# ---------------------------------------------------------------------------
# Paper metadata
# Keys match the paper short-keys used throughout macrocast documentation.
# filename: base name of the PDF in PDF_SOURCE_DIR.
# ---------------------------------------------------------------------------

PAPER_METADATA: dict[str, dict] = {
    "CLSS2022": {
        "filename": "10.1002jae.2910.pdf",
        "authors": "Coulombe, Leroux, Stevanovic, Surprenant",
        "year": 2022,
        "journal": "Journal of Applied Econometrics",
        "title": "How is Machine Learning Useful for Macroeconomic Forecasting?",
    },
    "C2024mrf": {
        "filename": "10.1002jae.3030.pdf",
        "authors": "Coulombe",
        "year": 2024,
        "journal": "Journal of Applied Econometrics",
        "title": "The Macroeconomy as a Random Forest",
    },
    "C2024tvp": {
        "filename": "10.1016j.ijforecast.2024.08.006.pdf",
        "authors": "Coulombe",
        "year": 2024,
        "journal": "International Journal of Forecasting",
        "title": "Time-Varying Parameters as Ridge Regressions",
    },
    "CLSS2021": {
        "filename": "10.1016j.ijforecast.2021.05.005.pdf",
        "authors": "Coulombe, Leroux, Stevanovic, Surprenant",
        "year": 2021,
        "journal": "International Journal of Forecasting",
        "title": "Macroeconomic Data Transformations Matter (MARX/MAF)",
    },
    "C2025bag": {
        "filename": "10.48550arXiv.2008.07063.pdf",
        "authors": "Coulombe",
        "year": 2025,
        "journal": "arXiv",
        "title": "To Bag Is to Prune",
    },
    "CGK2024": {
        "filename": "10.48550arXiv.2412.13076.pdf",
        "authors": "Coulombe, Gobel, Koop",
        "year": 2024,
        "journal": "arXiv",
        "title": "A Dual Interpretation of Machine Learning Forecasts",
    },
    "CGK2025": {
        "filename": "10.48550arXiv.2505.12422.pdf",
        "authors": "Coulombe, Gobel, Koop",
        "year": 2025,
        "journal": "arXiv",
        "title": "Opening the Black Box of Local Projections",
    },
    "CK2025": {
        "filename": "10.48550arXiv.2501.13222.pdf",
        "authors": "Coulombe, Koop",
        "year": 2025,
        "journal": "arXiv",
        "title": "AlbaMA: A Bayesian Machine Learning Approach to Macroeconomic Forecasting",
    },
    "C2025hnn": {
        "filename": "10.48550arXiv.2202.04146.pdf",
        "authors": "Coulombe",
        "year": 2025,
        "journal": "arXiv",
        "title": "The Neural Phillips Curve and Long-Run Inflation Expectations",
    },
    "CBRSS2022": {
        "filename": "10.2139ssrn.4278745.pdf",
        "authors": "Coulombe, Boivin, Rao, Stevanovic, Surprenant",
        "year": 2022,
        "journal": "SSRN",
        "title": "Anatomy of Out-of-Sample Gains",
    },
    "C2025ols": {
        "filename": "10.2139ssrn.5120847.pdf",
        "authors": "Coulombe",
        "year": 2025,
        "journal": "SSRN",
        "title": "OLS as Attention",
    },
    "C2024mfl": {
        "filename": "Maximally Forward-Looking Core Inflation.pdf",
        "authors": "Coulombe",
        "year": 2024,
        "journal": "Working Paper",
        "title": "Maximally Forward-Looking Core Inflation",
    },
    "CRMS2026": {
        "filename": "10.2139ssrn.4628462.pdf",
        "authors": "Coulombe, Rebucci, McCracken, Stevanovic",
        "year": 2026,
        "journal": "SSRN",
        "title": "SPPC: A Sparse Predictive Portfolio Construction Framework",
    },
    # Related non-Coulombe papers
    "HKKO2022": {
        "filename": "10.108007350015.2021.1990772.pdf",
        "authors": "Huber, Knaus, Kowal, Oelker",
        "year": 2022,
        "journal": "Journal of Business and Economic Statistics",
        "title": "Bayesian Time-Varying Parameter VAR Models",
    },
    "HLN2011": {
        "filename": "The Model Confidence Set.pdf",
        "authors": "Hansen, Lunde, Nason",
        "year": 2011,
        "journal": "Econometrica",
        "title": "The Model Confidence Set",
    },
}
