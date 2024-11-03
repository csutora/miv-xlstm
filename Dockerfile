FROM nvcr.io/nvidia/pytorch:24.09-py3

RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    htop \
    tmux\
    zsh && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip3 install --ignore-installed -r /tmp/requirements.txt

ENV WANDB_API_KEY=<YOUR_API_KEY>
ENV WANDB_BASE_URL=https://api.wandb.ai/

COPY entrypoint.sh /entrypoint
RUN chmod o+rx /entrypoint

WORKDIR /app
COPY miv-xlstm/ .
RUN chmod -R o+rwx /app

CMD ["/entrypoint"]