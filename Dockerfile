FROM continuumio/miniconda

COPY . .

RUN conda venv create -f venv.yml