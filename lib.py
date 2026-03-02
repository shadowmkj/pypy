from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
import os
from dotenv import load_dotenv
from pathlib import Path
from multiprocessing import Pool


load_dotenv()  # Load environment variables from .env file

converter = None


def init_worker():
    global converter
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.MPS),
        do_ocr=True,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def chunk_pages(total_pages, chunk_size):
    for i in range(1, total_pages + 1, chunk_size):
        yield (i, min(i + chunk_size - 1, total_pages))


def parse_chunk(pdf, page, count):
    if pdf.is_file() and pdf.match("*.pdf"):
        print(f"process {pdf}")
        doc = converter.convert(pdf, page_numbers=list(range(page, page + 20)))
        markdown = doc.document.export_to_markdown()
        print(f"Writing to {pdf.stem}.md")
        output_path = f"./markdowns/{pdf.stem}-{count}.md"
        with open(output_path, "w", encoding="utf-8") as pdf:
            pdf.write(markdown)
        print(f"Saved markdown to {output_path}")


def parse_pages(pdf_path, page_range):
    global converter
    start, end = page_range
    print(f"Processing pages {start}-{end}")
    doc = converter.convert(pdf_path, page_range=(start, end + 1))
    markdown = doc.document.export_to_markdown()
    output_path = f"./markdowns/{pdf_path.stem}-{start}-{end}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Saved {output_path}")
