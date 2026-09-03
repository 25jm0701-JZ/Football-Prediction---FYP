import pathlib
import re
import sys

import pypdf


def extract_text(path: pathlib.Path) -> tuple[pypdf.PdfReader, str]:
    reader = pypdf.PdfReader(str(path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return reader, text.replace("\x00", "")


def main() -> int:
    if len(sys.argv) > 1:
        path = pathlib.Path(sys.argv[1])
        reader, text = extract_text(path)
        print(f"==== {path.name} | pages {len(reader.pages)} ====")
        if len(sys.argv) > 2:
            for keyword in sys.argv[2:]:
                print(f"---- keyword: {keyword} ----")
                matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
                print(f"matches: {len(matches)}")
                for match in matches[:5]:
                    start = max(0, match.start() - 1200)
                    end = min(len(text), match.end() + 2200)
                    print(text[start:end])
                    print()
        else:
            print(text[:12000])
        return 0

    folder = pathlib.Path("要求/文献")
    skip = {"ratings.pdf", "reade_singleton_scorelines.pdf"}
    for path in sorted(folder.glob("*.pdf")):
        if path.name in skip:
            continue
        reader = pypdf.PdfReader(str(path))
        print(f"==== {path.name} | pages {len(reader.pages)} ====")
        for page_number, page in enumerate(reader.pages[:7], start=1):
            text = (page.extract_text() or "").replace("\x00", "")
            print(f"--- PAGE {page_number} ---")
            print(text[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
