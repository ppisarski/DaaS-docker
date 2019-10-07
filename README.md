# DaaS-docker
ML model in a Docker container

To build the image run:

    docker build -t ${IMAGE_NAME}:${VERSION} .

CHeck that the image was created and exists locally
    
    docker images
    
Tag your image

    docker tag ${IMAGE_ID} ${IMAGE_NAME}:${TAG}
    # or
    docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:${TAG}
    
Test the container locally

    docker run ${IMAGE_NAME}:${TAG}
