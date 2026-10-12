"""Verify the source data and feature-builder code before reproducing the search."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,default=here.parents[1])
    args=parser.parse_args()
    root=args.project_root.resolve()
    manifest=json.loads((here/"reference/input_manifest.json").read_text(encoding="utf-8"))
    errors=[]
    canonical=manifest.get("normalized_text_sha256",{})
    line_endings_only=[]
    for relative,expected in manifest["sha256"].items():
        path=root/relative
        if not path.is_file(): errors.append(f"Missing: {relative}")
        else:
            content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=expected:
                normalized=hashlib.sha256(content.replace(b"\r\n",b"\n")).hexdigest()
                if normalized==canonical.get(relative): line_endings_only.append(relative)
                else: errors.append(f"Changed: {relative}")
    if errors: raise SystemExit("Input mismatch:\n"+"\n".join(errors))
    print(f"Verified {len(manifest['sha256'])} source files against the reference run.")
    if line_endings_only:
        print(f"Accepted {len(line_endings_only)} files with CRLF/LF-only differences; all other bytes match.")


if __name__=="__main__": main()
