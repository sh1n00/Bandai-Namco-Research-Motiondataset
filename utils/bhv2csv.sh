#!/bin/sh

inputFolder="../dataset/Bandai-Namco-Research-Motiondataset-1/data"

for file in "$inputFolder"/*.bvh; do
    if [ -f "$file" ]; then
        echo "Found .bvh file: $file"
        /Users/riku-sh/.local/bin/poetry run bvh2csv --out "../data/raw/position" --position $file
        /Users/riku-sh/.local/bin/poetry run bvh2csv --out "../data/raw/rotation" --rotation $file
    fi
done
