from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions, PictureDescriptionApiOption
from docling.datamodel.base_models import InputFormat
import os

source = "./data/cn.pdf"  # file path or URL

picture_desc_api_option = PictureDescriptionApiOption(
    url="http://localhost:11434/api/generate",
    prompt="Describe the content of this image in a single paragraph.",
    params=dict(model="gemma:2b", temperature=0.2),
    headers={},  # No auth needed for local Ollama
    timeout=60
)

# Configure PdfPipelineOptions for OCR with Tesseract CLI
pipeline_options = PdfPipelineOptions(
    do_picture_description=True,
    picture_description_api_option=picture_desc_api_option,
    generate_picture_images=True,
    enable_remote_services=True,
    do_ocr=True,
    images_scale=2,
    ocr_options=TesseractCliOcrOptions(lang=["eng"])
)

# Initialize DocumentConverter with the configured options
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

doc = converter.convert(source).document

markdown = doc.export_to_markdown()
output_path = "./data/cn.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown)
print(f"Saved markdown to {output_path}")
