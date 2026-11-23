"""Execution identity for independently initialized portable runs."""
from pathlib import Path
import hashlib,json
def code_digest():
 root=Path(__file__).resolve().parent;h=hashlib.sha256()
 for p in sorted(root.rglob('*.py')):
  if '__pycache__' in p.parts or 'tests' in p.relative_to(root).parts:continue
  h.update(p.relative_to(root).as_posix().encode());h.update(hashlib.sha256(p.read_bytes()).digest())
 return h.hexdigest()
