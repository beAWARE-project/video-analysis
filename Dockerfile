FROM ppgiannak/obd:latest

COPY src/listener.py /usr/src/listener/
COPY src/dummy.py /usr/src/listener

WORKDIR /usr/src/listener/

CMD ["python3", "dummy.py"]
