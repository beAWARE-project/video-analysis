FROM ppgiannak/obd:latest

COPY src/video_listener.py /usr/src/listener/

WORKDIR /usr/src/listener/

ENV PYTHONPATH="/usr/local/lib/python3.5/site-packages/tensorflow/models/:/usr/local/lib/python3.5/site-packages/tensorflow/models/slim:${PYTHONPATH}"

CMD python3 video_listener.py
