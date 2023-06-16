#!/bin/sh

folder="../dataset/Bandai-Namco-Research-Motiondataset-1/data"

for file in "$folder"/*.bvh; do
    if [ -f "$file" ]; then
        echo "Found .bvh file: $file"
        /Users/riku-sh/.local/bin/poetry run bvh2csv --out ../position --position $file
        /Users/riku-sh/.local/bin/poetry run bvh2csv --out ../rotation --rotation $file
    fi
done
