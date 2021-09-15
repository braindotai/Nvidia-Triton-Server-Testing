FROM nvcr.io/nvidia/tritonserver:21.08-py3

RUN apt-get update -y && \
    apt-get install htop && \
    python3 -m pip install --upgrade pip && \
    adduser user

USER user

WORKDIR /home/user/src

# COPY /src .

EXPOSE 8000

EXPOSE 8001

EXPOSE 8002


CMD [ "bash" ]
# CMD [ "tritonserver", "--model-repository", "models"]