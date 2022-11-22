import os
import threading
import time
import logging
import torch
import numpy
import asyncio
import pickle
from io import BytesIO
from pathlib import Path

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.service.QueueService import QueueService

class YoloLegoDetector(LegoDetector):
    def __init__(self, model_path=os.path.join("lego_sorter_server", "analysis",
                                "detection", "models", "yolo_model", "yolov5_medium_extended.pt")):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()
        self.queue = None

    def initialize(self):
        if self.__initialized:
            raise Exception("[YoloLegoDetector] Already initialized")

        if not self.model_path.exists():
            message = f"[YoloLegoDetector] No model found in {str(self.model_path)}"
            logging.error(message)
            raise RuntimeError(message)

        start_time = time.time()
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=str(self.model_path))

        if torch.cuda.is_available():
            self.model.cuda()

        elapsed_time = time.time() - start_time
        logging.info("Loading model took {} seconds".format(elapsed_time))

        self.queue = QueueService()
        self.queue.subscribe('detect', self._detect_handler)
        self.__initialized = True
        self.queue.start()

    @staticmethod
    def xyxy2yxyx_scaled(xyxy):
        """
        returns (ymin, xmin, ymax, xmax)
        """
        return numpy.array([[coord[1], coord[0], coord[3], coord[2]] for coord in xyxy])

    @staticmethod
    def convert_results_to_common_format(results) -> DetectionResults:
        image_predictions = results.xyxyn[0].cpu().numpy()
        scores = image_predictions[:, 4]
        classes = image_predictions[:, 5].astype(numpy.int64) + 1
        boxes = YoloLegoDetector.xyxy2yxyx_scaled(image_predictions[:, :4])

        return DetectionResults(detection_scores=scores, detection_classes=classes, detection_boxes=boxes)

    def _detect_handler(self, channel, method, properties, body):
        image = numpy.load(BytesIO(body), allow_pickle=True)

        results = self.detect_lego(image)
        channel.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                body=pickle.dumps(results)
            )

    def detect_lego(self, image: numpy.ndarray) -> DetectionResults:
        if not self.__initialized:
            logging.info("[YoloLegoDetector] Initializing...")
            self.initialize()

        logging.info("[YoloLegoDetector] Detecting bricks...")
        start_time = time.time()
        results = self.model([image], size=image.shape[0])
        elapsed_time = 1000 * (time.time() - start_time)
        logging.info(
            f"[YoloLegoDetector] Detecting bricks took {elapsed_time} ms")

        return self.convert_results_to_common_format(results)
