FROM nvidia/cuda:10.0-devel-ubuntu18.04

#RUN yes | unminimize

RUN apt-get update && apt-get install -y wget bzip2
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes

RUN conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch
RUN pip install scikit-image tqdm pyyaml easydict future pip
RUN apt-get install unzip

COPY ./ /obow
RUN pip install -e /obow

WORKDIR /obow

# Test imports
RUN python -c ""
RUN python -c "import main_linear_classification"
RUN python -c "import main_obow"
RUN python -c "import main_semisupervised"
