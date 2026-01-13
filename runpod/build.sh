docker build --platform linux/amd64  -t statement-extractor-runpod .
docker tag statement-extractor-runpod neilellis/statement-extractor-runpod:v$1
docker push neilellis/statement-extractor-runpod:v$1
