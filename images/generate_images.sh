#!/bin/bash

# Loop through all .dot files in the current directory
for file in *.dot; do
  # Check if any .dot files exist
  [ -e "$file" ] || continue

  # Extract the base filename without the .dot extension
  filename="${file%.dot}"

  # Generate the PNG image using the dot command
  dot -Tpng "$file" -o "${filename}.png"

  echo "Generated ${filename}.png from ${file}"
done

# Loop through all .py files in the current directory
for file in *.py; do
  python3 $file
done