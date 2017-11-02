FROM ppgiannak/obd:latest

COPY src/listener.py /usr/src/listener/

WORKDIR /usr/src/dummy/

CMD ["python3", "dummy.py"]
