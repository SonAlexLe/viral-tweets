docker run -it --mount type=bind,source="$(pwd)"/,target=/root/data $(docker build -q .) twint --near "New York" -o data/output.json --json --limit 100 --stats

