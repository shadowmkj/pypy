from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from pathlib import Path
from multiprocessing import Pool

from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter


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


def chunk_pages(total_pages, chunk_size, skip=0):
    """Yield (start, end) page ranges.

    `skip` defines how many pages to skip from the beginning.
    For example, total_pages=10, chunk_size=3, skip=2 ->
    (3,5), (6,8), (9,10).
    """

    start_page = max(1, skip + 1)
    for i in range(start_page, total_pages + 1, chunk_size):
        yield (i, min(i + chunk_size - 1, total_pages))


def strip_hyperlinks(input_path: Path) -> Path:
    """Return a copy of the PDF with clickable hyperlinks removed.

    Only link annotations (/Subtype /Link) are removed. Visible URL text
    remains unchanged.
    """

    output_path = input_path.with_name(input_path.stem + "_nolinks" + input_path.suffix)

    # Simple cache: reuse the cleaned file if it exists and is up-to-date.
    if (
        output_path.exists()
        and output_path.stat().st_mtime >= input_path.stat().st_mtime
    ):
        return output_path

    reader = PdfReader(str(input_path))
    writer = PdfWriter()

    for page in reader.pages:
        if "/Annots" in page:
            annots = page["/Annots"]
            new_annots = []
            for annot_ref in annots:
                annot = annot_ref.get_object()
                subtype = annot.get("/Subtype")
                # Drop only link annotations.
                if subtype == "/Link":
                    continue
                new_annots.append(annot_ref)

            if new_annots:
                page["/Annots"] = new_annots
            else:
                # No annotations remain on this page.
                page.pop("/Annots")

        writer.add_page(page)

    # Preserve document metadata when present.
    if reader.metadata:
        writer.add_metadata(reader.metadata)

    with output_path.open("wb") as f:
        writer.write(f)

    return output_path


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
