# Containerization

To up docker container run the following command inside current directory:  

```bash
docker build . -t ml_ops:containerization \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g)
```

```bash
docker run -d -it --rm \
    --name star \
    --mount type=bind,source="$(pwd)"/volumes,target=/starspace/volumes \
    --mount type=bind,source="$(pwd)"/model,target=/starspace/model \
    --env input_file_name="starspace_input_file.txt" \
    --env output_file_name="checkpoint" \
    ml_ops:containerization
```
