FROM python:3.10

WORKDIR /etc/case_study

COPY . /etc/case_study/.

RUN  pip install --upgrade pip && pip --no-cache-dir install -r /etc/case_study/requirements.txt

ENV PYTHONPATH $PYTHONPATH:$PATH:/etc/case_study/src/

ENV PATH /opt/conda/envs/env/bin:$PATH

ENV PROJECT_PATH /etc/case_study/src/

EXPOSE 1234

ENTRYPOINT gunicorn -b 0.0.0.0:1234 -k uvicorn.workers.UvicornWorker main:app --threads 2 --workers 1 --timeout 1000 --graceful-timeout 30
