import logging
import gc
import time
import shutil
from pathlib import Path
from typing import List

from pypdf import PdfReader, PdfWriter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend 

from src.utils.io import write_text

OUTPUT_DIR = Path("data/processed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def split_pdf_for_processing(input_pdf: Path, output_dir: Path, max_pages: int = 50) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(input_pdf))
    total = len(reader.pages)
    chunks = []

    for i in range(0, total, max_pages):
        writer = PdfWriter()
        start, end = i, min(i + max_pages, total)
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        chunk_path = output_dir / f"{input_pdf.stem}_part_{i//max_pages + 1}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append(chunk_path)
        logging.info(f"✓ Chunk: {chunk_path.name} (стр. {start+1}-{end})")

    reader.stream.close()
    del reader
    gc.collect()
    return chunks


def load_pdf(path: str, chunk_size: int = 50) -> Path:
    pdf_path = Path(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{pdf_path.stem}.md"

    if output_file.exists():
        size = output_file.stat().st_size
        if size == 0 or size < 100:  
            logging.warning(f"The existing markdown is empty ({size} байт) — delete and reparse: {output_file}")
            output_file.unlink()
        else:
            logging.info(f"✓ Markdown already exist: {output_file}")
            return output_file


def _load_pdf_simple(pdf_path: Path, output_file: Path) -> Path:
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.generate_parsed_pages = False
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend  
                )
            }
        )

        conv_result = converter.convert(source=str(pdf_path))

        if conv_result.status == "success":
            markdown_content = conv_result.document.export_to_markdown()
            write_text(markdown_content, output_file)
            logging.info(f"✓ Saved: {output_file} ({len(markdown_content):,} characters)")
        else:
            logging.error(f"Error in conversion: {conv_result.errors}")
            write_text("", output_file)
    except Exception as e:
        logging.exception(f"Crash in simple: {e}")
        write_text("", output_file)

    return output_file


def _load_pdf_chunked(pdf_path: Path, output_file: Path, chunk_size: int = 50) -> Path:
    logging.warning(f"🔥 Big file ({chunk_size} page./chunk), processing...")

    temp_dir = OUTPUT_DIR / "temp_chunks"
    chunk_paths = split_pdf_for_processing(pdf_path, temp_dir, max_pages=chunk_size)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.generate_parsed_pages = False
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend  
            )
        }
    )

    docling_docs = []
    for idx, chunk_path in enumerate(chunk_paths, 1):
        logging.info(f"Chunk {idx}/{len(chunk_paths)}")
        try:
            result = converter.convert(source=str(chunk_path))
            if result.status == "success":
                docling_docs.append(result.document)
            else:
                logging.error(f"Error of chunk: {result.errors}")
        except Exception as e:
            logging.exception(f"Crash of chunk {chunk_path.name}: {e}")
        finally:
            if chunk_path.exists():
                chunk_path.unlink()
            gc.collect()
            time.sleep(0.05)  

    if docling_docs:
        try:
            from docling_core.types.doc.document import DoclingDocument
            merged_doc = DoclingDocument.concatenate(docling_docs)
            markdown_content = merged_doc.export_to_markdown()
        except Exception:
            markdown_parts = [doc.export_to_markdown() for doc in docling_docs]
            markdown_content = "\n\n".join(markdown_parts)

        write_text(markdown_content, output_file)
        logging.info(f"✓ Saved: {output_file} ({len(markdown_content):,} characters)")
    else:
        write_text("", output_file)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return output_file