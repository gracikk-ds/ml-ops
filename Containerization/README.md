# Containerization

To up docker container run the following command inside current directory:  

```bash
docker build . -t ml_ops:containerization
```

```bash
docker run -d -it --rm \
    --name star \
    --mount type=bind,source="$(pwd)"/volumes,target=/volumes \
    --mount type=bind,source="$(pwd)"/model,target=/model \
    --env input_file_name="starspace_input_file.txt" \
    --env output_file_name="checkpoint" \
    ml_ops:containerization
```
