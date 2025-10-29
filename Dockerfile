FROM continuumio/miniconda3

WORKDIR /usr/src/app

# System build dependencies needed for some pip packages (e.g., zstd, torch-scatter)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	   build-essential \
	   pkg-config \
	   libzstd-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY . .

# The conda environment should be from the dgx (!!NOT from your PC)

RUN conda env create -f environment.yaml

SHELL ["/bin/bash", "-c"]

RUN conda init bash

RUN chmod +x additional-softwares.sh

# Replace <env-name> with the name of your conda environment
RUN /bin/bash -c "source activate crossppi && ./additional-softwares.sh"
RUN /bin/bash -c "source activate crossppi && pip install git+https://github.com/facebookresearch/esm.git"

EXPOSE 5060

# Replace <env-name> with the name of your conda environment
CMD [ "bash", "-lc", "source activate crossppi  && exec gunicorn server:app -b 0.0.0.0:5060 --workers=1 --pythonpath /usr/src/app/src" ]
