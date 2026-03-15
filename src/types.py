from typing import TypedDict, List, Union, Literal, Dict

class ChunkMetadata(TypedDict):
    page_index: int
    pdf_sha1: str
    chunk_index: int

class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata 

class PageMetadata(TypedDict):
    page_no: int
    one_based: bool

class Page(TypedDict):
    pdf_sha1: str
    pdf_path: str
    page_index: int
    text: str
    metadata: PageMetadata

class Embedding(TypedDict):
    vector: List[float]
    text: str
    metadata: ChunkMetadata

class VectorPoint(TypedDict):
    vector: List[float]
    text: str
    metadata: dict

class References(TypedDict):
    pdf_sha1: str
    page_index:int

class Answer(TypedDict):
    value: Union[str, int, float]
    references: List[References]

class Question(TypedDict):
    text: str
    kind: Literal['names', 'number', 'boolean']