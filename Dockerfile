FROM ppgiannak/obd:latest

COPY src/listener.py /usr/src/listener/

WORKDIR /usr/src/listener/

CMD ["python3", "dummy.py"]
