import logging
import os
import time

import tensorflow as tf
import numpy as np
import pickle
import threading
import PIL.Image

from io import BytesIO

from typing import List

from tensorflow import keras
from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.classifiers.LegoClassifier import LegoClassifier
from lego_sorter_server.analysis.classification.toolkit.transformations.simple import Simple
from lego_sorter_server.service.QueueService import QueueService

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class KerasClassifier(LegoClassifier):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis", "classification", "models",
                                               "keras_model", "447_classes.h5")):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.initialized = False
        self.size = (224, 224)
        self.queue = QueueService()

    def load_model(self):
        self.model = keras.models.load_model(self.model_path)
        self.queue.subscribe('classify', self._classify_handler)
        self.queue.start()
        self.initialized = True

    def _classify_handler(self, channel, method, properties, body):
        logging.info(f'[KerasClassifier] classifying...')
        image = PIL.Image.fromarray(np.load(BytesIO(body), allow_pickle=True))

        results = self.predict(image)
        channel.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            body=pickle.dumps(results)
        )

    def predict(self, image: Image) -> ClassificationResults:
        if not self.initialized:
            self.load_model()

        start_time = time.time()
        transformed = Simple.transform(image, self.size[0])
        image_array = np.array(transformed)
        image_array = np.expand_dims(image_array, axis=0)
        processing_elapsed_time_ms = 1000 * (time.time() - start_time)

        predictions = self.model(image_array)

        predicting_elapsed_time_ms = 1000 * \
            (time.time() - start_time) - processing_elapsed_time_ms

        logging.info(f"[KerasClassifier] Preparing images took {processing_elapsed_time_ms} ms, "
                     f"when predicting took {predicting_elapsed_time_ms} ms.")

        indices = [int(np.argmax(values)) for values in predictions]
        classes = [self.class_names[index] for index in indices]
        scores = [float(prediction[index])
                  for index, prediction in zip(indices, predictions)]

        return ClassificationResults(classes, scores)
