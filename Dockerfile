FROM ppgiannak/obd:latest

WORKDIR /usr/src/dummy/

CMD python3 listener2.py
