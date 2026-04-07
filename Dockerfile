FROM vastai/pytorch:latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./PyDiff/requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -U numpy

COPY . .

WORKDIR /app/PyDiff

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN apt install -y wget g++

######################

RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

WORKDIR /app/BasicSR-light
RUN pip install -r /app/BasicSR-light/requirements.txt
ENV BASICSR_EXT=True 
RUN python3 setup.py develop
WORKDIR /app/PyDiff
# PyDiff pins numpy<1.21, but Python 3.10 in the base image has no wheel for that range.
RUN sed -i 's/^numpy<1\.21.*/numpy==1.23.5/' /app/PyDiff/requirements.txt
RUN pip install -r /app/PyDiff/requirements.txt
ENV BASICSR_EXT=True 
RUN python3 setup.py develop
RUN pip install -U numpy

######################

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


# # install miniconda
# ENV PATH="/root/miniconda3/bin:$PATH"
# RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

# RUN ls
# # create conda environment
# RUN conda init bash && . ~/.bashrc && conda create -n PyDiff python=3.7 && conda activate PyDiff && conda install -y pytorch==1.13.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch

    
# WORKDIR /app/BasicSR-light
# RUN conda init bash && . ~/.bashrc && conda activate PyDiff && pip install -r /app/BasicSR-light/requirements.txt
# ENV BASICSR_EXT=True 
# RUN conda init bash && . ~/.bashrc && conda activate PyDiff && python setup.py develop
# WORKDIR /app/PyDiff
# RUN conda init bash && . ~/.bashrc && conda activate PyDiff && pip install -r /app/PyDiff/requirements.txt
# ENV BASICSR_EXT=True 
# RUN conda init bash && . ~/.bashrc && conda activate PyDiff && python setup.py develop

# WORKDIR /app/PyDiff