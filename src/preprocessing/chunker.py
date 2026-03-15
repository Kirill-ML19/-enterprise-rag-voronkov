from typing import Generator, List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from src.types import Chunk

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=95  
)

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3')
    ],
    strip_headers=False,
    return_each_line=False
)

def chunk_md(
    md_text: str,
    pdf_sha1: str,
    batch_size: int = 10,
) -> Generator[List[Chunk], None, None]:
    batch: List[Chunk] = []
    headers_docs = header_splitter.split_text(md_text)

    chunk_idx = 0
    page_index = 0
    for doc in headers_docs:
        section_text = doc.page_content
        metadata = doc.metadata

        semantic_chunks = semantic_splitter.split_text(section_text)

        for chunk_text in semantic_chunks:
            batch.append({
                'text': chunk_text,
                'metadata': {
                    'pdf_sha1': pdf_sha1,
                    'chunk_index': chunk_idx,
                    'chunk_id': f"{pdf_sha1}_{chunk_idx}", 
                    'page_index': page_index,
                    'headers': metadata,
                    'type': 'chunk', 
                }
            })
            chunk_idx += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        page_index+=1
    
    if batch:
        yield batch    