### TO build docker image run the following command from the root of the project:
```bash
docker build \
-f docker/service_image/Dockerfile \
-t grac20101/aleksandr-gordeev:latest \
--build-arg USER_ID=$(id -u) \
--build-arg GROUP_ID=123 \
.
```

