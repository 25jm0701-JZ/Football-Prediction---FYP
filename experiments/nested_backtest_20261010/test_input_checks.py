"""Input audit accepts only line-ending changes, never changed data values."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import check_inputs


class InputChecks(unittest.TestCase):
    def check(self, current):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=b"date,result\r\n2025-08-01,H\r\n"
            (root/"matches.csv").write_bytes(current)
            reference=root/"experiment/reference"; reference.mkdir(parents=True)
            manifest={"sha256":{"matches.csv":hashlib.sha256(original).hexdigest()},
                "normalized_text_sha256":{"matches.csv":hashlib.sha256(original.replace(b"\r\n",b"\n")).hexdigest()}}
            (reference/"input_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
            capture=io.StringIO()
            with patch.object(check_inputs,"__file__",str(root/"experiment/check_inputs.py")), \
                 patch("sys.argv",["check_inputs.py","--project-root",str(root)]),redirect_stdout(capture):
                check_inputs.main()
            return capture.getvalue()

    def test_original_bytes_pass(self):
        output=self.check(b"date,result\r\n2025-08-01,H\r\n")
        self.assertIn("Verified 1 source files",output)
        self.assertNotIn("Accepted",output)

    def test_git_line_ending_conversion_passes(self):
        self.assertIn("Accepted 1 files",self.check(b"date,result\n2025-08-01,H\n"))

    def test_changed_match_result_is_rejected(self):
        with self.assertRaisesRegex(SystemExit,"Changed: matches.csv"):
            self.check(b"date,result\n2025-08-01,A\n")


if __name__=="__main__": unittest.main(verbosity=2)
