FROM python:3.6

RUN mkdir usr/src/app
WORKDIR usr/src/app

COPY . .
RUN python -m pip install -r requirements.txt

EXPOSE 4000
CMD [ "python", "app.py" ]