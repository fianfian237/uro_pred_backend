FROM python:3.7

RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir pyparsing pydot

RUN mkdir usr/src/app
WORKDIR usr/src/app

COPY requirements.txt /tmp/
RUN python -m pip install -r /tmp/requirements.txt

COPY . .

#EXPOSE 4000 default port flask=5000
#CMD [ "python", "wsgi.py" ]