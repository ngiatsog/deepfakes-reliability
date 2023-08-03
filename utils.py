import os
import uuid

from pathlib import Path


def read_csv(path, delim="\t", types=None, with_headers=False):
    with open(path, "r", encoding="utf8") as f: 
        headers = None
        rows = []        
        if with_headers:
            line = f.readline()
            headers = line.rstrip().split(delim)
        for line in f:
            row = line.rstrip().split(delim)
            if types is not None:
                row = [cast(item) for item, cast in zip(row, types)]
            rows.append(row)
        if with_headers:
            return rows, headers
        else:
            return rows

def write_csv(path, rows, delim="\t", headers=None):
    with open(path, "w", encoding="utf8") as f:
        if headers is not None:
            line = delim.join([str(header) for header in headers])
            f.write(line)
            f.write("\n")
            
        for row in rows:
            line = delim.join([str(item) for item in row])
            f.write(line)
            f.write("\n")

def image_iter(path):  
    for base, _, files in os.walk(path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                yield Path(base) / file


def random_filename():
    return str(uuid.uuid4())