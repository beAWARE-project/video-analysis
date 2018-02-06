FROM ppgiannak/obd:latest

COPY src/video_listener.py /usr/src/listener/
COPY src/video_analyzer.py /usr/src/listener/
COPY src/model/label_map.pbtxt /usr/src/listener/model/

WORKDIR /usr/src/listener/model/
RUN wget -O frozen_inference_graph.pb http://object-store-app.eu-gb.mybluemix.net/objectStorage?file=frozen_inference_graph.pb

WORKDIR /usr/src/listener/

ENV PYTHONPATH="/usr/local/lib/python3.5/site-packages/tensorflow/models/:/usr/local/lib/python3.5/site-packages/tensorflow/models/slim:${PYTHONPATH}"

CMD python3 video_listener.py
