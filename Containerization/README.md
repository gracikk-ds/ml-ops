# Containerization

To up docker container run the following command inside current directory:

```bash
docker run -d -it --rm \
    --name star \
    --mount type=bind,source="$(pwd)"/volumes,target=/volumes \
    --mount type=bind,source="$(pwd)"/model,target=/model \
    --env input_file=/volumes/starspace_input_file.txt \
    ml_ops:test
```