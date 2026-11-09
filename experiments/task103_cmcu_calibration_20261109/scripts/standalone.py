"""Embed figure PNGs so the detailed report opens without a server or CDN."""
import argparse
import base64
from pathlib import Path
import re


def build(source, destination):
    source, destination = Path(source), Path(destination)
    text = source.read_text()
    def embed(match):
        relative = match.group(1)
        if relative.startswith('data:'): return match.group(0)
        data = base64.b64encode((source.parent/relative).read_bytes()).decode()
        return '<img src="data:image/png;base64,'+data+'"'
    text = re.sub(r'<img src="([^"]+)"', embed, text)
    destination.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('destination', type=Path)
    args = parser.parse_args(); build(args.source, args.destination)
