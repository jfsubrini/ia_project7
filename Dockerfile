FROM python:3.9
WORKDIR /app
COPY . /app
RUN chmod +x start.sh
RUN pip install -r requirements.txt
CMD ["./start.sh"]
