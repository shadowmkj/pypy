import sys
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from lib import chunk_pages, parse_pages, init_worker

load_dotenv()  # Load environment variables from .env file

if __name__ == "__main__":
    source_folder = Path("./data")
    files = [p for p in source_folder.iterdir()]
    count = 0

    for f in files:
        if f.stem == "book":
            total_pages = 50
            chunk_size = 5
            page_chunks = list(chunk_pages(total_pages, chunk_size))
            with ProcessPoolExecutor(
                max_workers=2, initializer=init_worker
            ) as executor:
                futures = []
                for count, (start, end) in enumerate(page_chunks, 1):
                    futures.append(
                        executor.submit(
                            parse_pages,
                            f,
                            (start, end),
                        )
                    )

                for f in futures:
                    f.result()

    print("EXITING")
    sys.exit()
