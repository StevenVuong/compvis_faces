FROM python:3.7

WORKDIR /usr/src/app

COPY . .

RUN apt-get update 
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./frame_extractor.py" ]