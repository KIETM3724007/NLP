import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_sentences(text):
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
    return nltk.sent_tokenize(" ".join(text) if isinstance(text, list) else text)

# Code của Nghĩa
# def align_abstract_to_sections(sections, abstract_sents):
#     aligned = [[] for _ in sections]

#     for sent in abstract_sents:
#         idx = min(range(len(sections)), key=lambda i: abs(len(sections[i]) - len(sent)))
#         aligned[idx].append(sent)

#     return [" ".join(s) for s in aligned]

# Code của Nghĩa
# def clean_article(text: str) -> str:
#     """Làm sạch LaTeX và citation trong article."""
#     text = text.replace("\\n", "\n")
#     text = re.sub(r"@xmath\d+", "", text)
#     text = re.sub(r"@xcite", "", text)
#     return text


def align_abstract_to_sections(sections, abstract_sents):
    if not sections or not abstract_sents:
        return [""] * len(sections)

    # Corpus = sections + abstract sentences
    corpus = sections + abstract_sents

    vectorizer = TfidfVectorizer(min_df=1, max_df=1.0, stop_words="english")  # BẮT BUỘC

    X = vectorizer.fit_transform(corpus)
    sec_vecs = X[: len(sections)]
    abs_vecs = X[len(sections) :]

    aligned = [[] for _ in sections]

    for i, abs_v in enumerate(abs_vecs):
        sims = cosine_similarity(abs_v, sec_vecs)[0]
        best_idx = sims.argmax()
        aligned[best_idx].append(abstract_sents[i])

    return [" ".join(s) for s in aligned]


def clean_article(text: str) -> str:
    if not text:
        return ""

    # ----------- BẢO TOÀN TOKEN ĐẶC BIỆT (nếu bạn có) -----------
    special_tokens = ["<S>", "</S>", "<SENT/>", "<SCTN/>", "<EI/>"]
    token_map = {tok: f"__SPECIAL_{i}__" for i, tok in enumerate(special_tokens)}
    for tok, ph in token_map.items():
        text = text.replace(tok, ph)

    # ----------- NORMALIZE -----------
    text = text.replace("\\n", "\n")

    # ----------- REMOVE COMMENTS -----------
    text = re.sub(r"%.*", " ", text)

    # ----------- REMOVE DISPLAY MATH -----------
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.S)

    # ----------- REMOVE INLINE MATH -----------
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\\([^)]*\\\)", " ", text)

    # ----------- REMOVE ENVIRONMENTS -----------
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.S)

    # ----------- REMOVE FIGURES / TABLES / INCLUDE -----------
    text = re.sub(r"\\includegraphics\{.*?\}", " ", text)
    text = re.sub(r"\\caption\{.*?\}", " ", text)

    # ----------- REMOVE REFERENCES & CITATIONS -----------
    text = re.sub(r"\\cite\{.*?\}", " ", text)
    text = re.sub(r"\\ref\{.*?\}", " ", text)
    text = re.sub(r"\\label\{.*?\}", " ", text)
    text = re.sub(r"\\pageref\{.*?\}", " ", text)

    # ----------- REMOVE STRUCTURAL COMMANDS -----------
    text = re.sub(
        r"\\(section|subsection|subsubsection|paragraph)\*?\{.*?\}", " ", text
    )

    # ----------- REMOVE GENERIC COMMANDS -----------
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{.*?\})?", " ", text)

    # ----------- REMOVE BRACES & SCRIPTS -----------
    text = re.sub(r"[\^_]\{.*?\}", " ", text)
    text = re.sub(r"\{[^{}]*\}", " ", text)

    # ----------- CLEAN LEFTOVER SYMBOLS -----------
    text = re.sub(r"[\\$]", " ", text)

    # ----------- NORMALIZE WHITESPACE -----------
    text = re.sub(r"\s+", " ", text).strip()

    # ----------- RESTORE SPECIAL TOKENS -----------
    for tok, ph in token_map.items():
        text = text.replace(ph, tok)

    return text


def segment_article(article: str, max_chars: int = 2000):
    """Chia article thành các page nhỏ."""
    text = clean_article(article)
    paragraphs = text.split("\n\n")

    pages, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            pages.append(current.strip())
            current = para + "\n\n"
    if current:
        pages.append(current.strip())

    return pages
