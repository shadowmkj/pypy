import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from lib import chunk_pages, init_worker, parse_pages, strip_hyperlinks

load_dotenv()  # Load environment variables from .env file

if __name__ == "__main__":
    source_folder = Path("./data")
    file_name = sys.argv[1] if len(sys.argv) > 1 else "book"
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    skip = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    files = [p for p in source_folder.iterdir()]
    count = 0

    for f in files:
        if f.stem == file_name:
            cleaned_pdf = strip_hyperlinks(f)
            total_pages = pages
            page_chunks = list(chunk_pages(total_pages, chunk_size, skip=skip))
            with ProcessPoolExecutor(
                max_workers=2, initializer=init_worker
            ) as executor:
                futures = []
                for count, (start, end) in enumerate(page_chunks, 1):
                    futures.append(
                        executor.submit(
                            parse_pages,
                            cleaned_pdf,
                            (start, end),
                        )
                    )

                for f in futures:
                    f.result()

    print("EXITING")
    sys.exit()
