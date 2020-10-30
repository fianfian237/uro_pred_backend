FROM python:3.6

RUN apt-get update \
  && apt-get install -y --no-install-recommends graphviz \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir pyparsing pydot

RUN mkdir usr/src/app
WORKDIR usr/src/app

COPY . .
RUN python -m pip install -r requirements.txt

EXPOSE 4000
CMD [ "python", "app.py" ]