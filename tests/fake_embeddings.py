"""A deterministic stand-in for the embedding model: texts that share words get close vectors."""

import math
import re
import zlib

DIMENSIONS = 384
# Words that carry no meaning for retrieval, so they do not pull unrelated texts together.
STOP_WORDS = frozenset(
    "a an the of in on at to for and or is was were what which how did do does by per with its "
    "it as be this that from each".split()
)


def embed(text: str) -> list[float]:
    """A bag-of-words vector: each word adds to one of 384 positions, then the vector is scaled."""
    vector = [0.0] * DIMENSIONS
    for word in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")):
        if word in STOP_WORDS:
            continue
        stem = word.removesuffix("s")  # "prices" and "price" count as the same word
        vector[zlib.crc32(stem.encode()) % DIMENSIONS] += 1.0
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


class FakeEmbeddings:
    """Has the two methods the code uses, and counts what it was asked to embed."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents += texts
        return [embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return embed(text)
