import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_cleaner import clean_pdf_text

PDF_DIR = Path("data/raw/pdf")
OUTPUT_DIR = Path("data/processed")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

CHUNK_SIZE = 100
PROCESSING_TIMEOUT = 3600 


async def read_file_async(path: Path) -> str:
    def _read():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return await asyncio.to_thread(_read)


async def write_file_async(path: Path, content: str):
    def _write():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    await asyncio.to_thread(_write)


async def process_pdf(pdf_file: str) -> Optional[Path]:
    pdf_path = PDF_DIR / pdf_file
    logging.info(f"Start: {pdf_file}")
    
    try:
        md_path = await asyncio.wait_for(
            asyncio.to_thread(load_pdf, str(pdf_path), chunk_size=CHUNK_SIZE),
            timeout=PROCESSING_TIMEOUT
        )
        
        if not md_path.exists() or md_path.stat().st_size == 0:
            logging.warning(f"Empty markdown for {pdf_file}")
            return md_path

        logging.info(f"Text cleaning: {md_path.name}")
        content = await read_file_async(md_path)
        cleaned_content = clean_pdf_text(content)
        await write_file_async(md_path, cleaned_content)

        logging.info(f"Ready: {pdf_file}")
        return md_path
        
    except asyncio.TimeoutError:
        logging.error(f"Timeout ({PROCESSING_TIMEOUT}s): {pdf_file}")
        quarantine_dir = OUTPUT_DIR / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)
        shutil.copy(pdf_path, quarantine_dir / pdf_file)
        return None
    except Exception as e:
        logging.exception(f"Error: {pdf_file}: {e}")
        return None


async def pdf_generator():
    if not PDF_DIR.exists():
        logging.error(f"Folder not found: {PDF_DIR}")
        return
        
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    logging.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        result = await process_pdf(pdf_file)
        if result:
            yield result


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    async for md_file in pdf_generator():
        logging.info(f"Ready: {md_file}")
    
    logging.info("All tasks finished")


if __name__ == "__main__":
    asyncio.run(main())