FROM python:3.7

ADD . /opt/marksheetreader
WORKDIR /opt/marksheetreader
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "./main.py"]
