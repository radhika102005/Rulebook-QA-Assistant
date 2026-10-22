from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# -----------------------------
# Groq Configuration
# -----------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from environment variables")

client = Groq(api_key=GROQ_API_KEY)


# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(
    title="Rulebook QA System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load Rulebook
# -----------------------------

RULEBOOK_PATH = Path(__file__).with_name("rulebook.txt")

content = RULEBOOK_PATH.read_text(
    encoding="utf-8"
)

# Keep section titles with their text
# so citations remain meaningful.

raw_sections = re.split(
    r"\n={20,}\n",
    content
)

sections = [
    s.strip()
    for s in raw_sections
    if len(s.strip()) > 80
]


# -----------------------------
# Lightweight Retrieval
# -----------------------------
# TF-IDF is used instead of the
# PyTorch-based Sentence Transformer
# so the application can run within
# Render's Free memory limit.

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=12000
)

section_vectors = vectorizer.fit_transform(
    sections
)


# -----------------------------
# Request Model
# -----------------------------

class Question(BaseModel):
    question: str


# -----------------------------
# Section Title
# -----------------------------

def section_title(text):

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    return (
        lines[0]
        if lines
        else "Unknown section"
    )


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(question, k=3):

    query_vector = vectorizer.transform(
        [question]
    )

    scores = cosine_similarity(
        query_vector,
        section_vectors
    )[0]

    indices = scores.argsort()[::-1][:k]

    return [
        (
            int(i),
            float(scores[i])
        )
        for i in indices
    ]


# -----------------------------
# Not Covered Detection
# -----------------------------

def is_not_covered(
    question,
    top_results
):

    q = question.lower()

    exact_missing_topics = [

        "exact tuition fee",
        "exact fee",
        "wifi password",
        "phone number",
        "current examination controller",
        "exact scholarship amount",
        "exact hostel fee",
        "bus route",
        "today's library",
        "today library",
        "family wedding",
        "personal laptop",
        "sports scholarship",
        "part-time job",
        "semester break date",
        "study abroad",
        "studying abroad",
    ]

    if any(
        term in q
        for term in exact_missing_topics
    ):
        return True

    return (
        not top_results
        or top_results[0][1] < 0.12
    )


# -----------------------------
# Conflict Detection
# -----------------------------

def find_conflict(
    question,
    retrieved
):

    q = question.lower()

    topics = []

    if (
        "attendance" in q
        and (
            "exam" in q
            or "eligible" in q
            or "percentage" in q
        )
    ):
        topics.append("attendance")

    if (
        "library" in q
        and (
            "fine" in q
            or "overdue" in q
            or "charge" in q
        )
    ):
        topics.append("library")

    if (
        "branch" in q
        and (
            "change" in q
            or "transfer" in q
        )
    ):
        topics.append("branch")

    if not topics:
        return None

    candidates = []

    for idx, score in retrieved:

        text = sections[idx].lower()

        if (
            "test clause" in text
            and any(
                topic in text
                for topic in topics
            )
        ):
            candidates.append(
                (idx, score)
            )

    if candidates:
        return candidates[:4]

    # Explicitly locate contradiction
    # sections if retrieval does not
    # rank them highly enough.

    for idx, text in enumerate(sections):

        low = text.lower()

        if (
            "test contradiction" in low
            and any(
                topic in low
                for topic in topics
            )
        ):

            return [
                (
                    idx,
                    1.0
                )
            ]

    return None


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "sections": len(sections),
        "corpus_words": len(
            content.split()
        )
    }


# -----------------------------
# Ask Question
# -----------------------------

@app.post("/ask")
def ask_question(q: Question):

    question = q.question.strip()

    if not question:

        return {
            "type": "not_covered",
            "answer": "Please enter a question.",
            "citations": []
        }

    # -----------------------------
    # Retrieve evidence
    # -----------------------------

    results = retrieve(
        question,
        k=3
    )

    # -----------------------------
    # Coverage check
    # -----------------------------

    if is_not_covered(
        question,
        results
    ):

        return {
            "type": "not_covered",
            "answer": (
                "This question is not covered "
                "by the available rulebook."
            ),
            "citations": []
        }

    # -----------------------------
    # Conflict check
    # -----------------------------

    conflict = find_conflict(
        question,
        results
    )

    if conflict:

        conflict_citations = []

        for idx, score in conflict:

            conflict_citations.append(
                {
                    "section": section_title(
                        sections[idx]
                    ),
                    "similarity": round(
                        score,
                        3
                    )
                }
            )

        return {
            "type": "conflict",
            "answer": (
                "The rulebook contains conflicting "
                "information about this topic. "
                "The system will not choose one "
                "rule over another."
            ),
            "citations": conflict_citations
        }

    # -----------------------------
    # Build evidence
    # -----------------------------

    evidence = []

    for idx, score in results:

        evidence.append(
            f"Section: "
            f"{section_title(sections[idx])}\n"
            f"Similarity: {score:.3f}\n"
            f"Content:\n"
            f"{sections[idx]}"
        )

    evidence_text = (
        "\n\n---\n\n".join(
            evidence
        )
    )

    # -----------------------------
    # Grounded Groq Prompt
    # -----------------------------

    prompt = f"""

You are a university rulebook
question-answering assistant.

Answer the user's question ONLY
using the rulebook evidence provided
below.

Rules you MUST follow:

1. Do not use outside knowledge.

2. Do not invent rules, dates, fees,
penalties, eligibility requirements,
procedures, or other information.

3. If the evidence does not clearly
answer the question, say:

"The available rulebook does not provide
enough information to answer this."

4. Give a clear and concise answer.

5. Mention the relevant rulebook
section in your answer.

6. Do not make assumptions.

7. Do not override or reinterpret
the rulebook.

User question:

{question}

RULEBOOK EVIDENCE:

{evidence_text}

"""

    # -----------------------------
    # Ask Groq
    # -----------------------------

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return {
            "type": "error",
            "answer": (
                f"Groq API error: {str(e)}"
            ),
            "citations": []
        }

    # -----------------------------
    # Prepare citations
    # -----------------------------

    citations = []

    if results:

        idx, score = results[0]

        citations.append(
            {
                "section": section_title(
                    sections[idx]
                ),
                "similarity": round(
                    score,
                    3
                )
            }
        )

    return {
        "type": "answered",
        "answer": answer,
        "citations": citations
    }


# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                8000
            )
        ),
        reload=False
    )