import os
import threading
import time
import logging
import torch
import numpy
import asyncio
from io import BytesIO
from pathlib import Path

from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.service.QueueService import QueueService

class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(
                        ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class YoloLegoDetector(LegoDetector, metaclass=ThreadSafeSingleton):
    DEFAULT_PATH = os.path.join("lego_sorter_server", "analysis",
                                "detection", "models", "yolo_model", "yolov5_medium_extended.pt")

    def __init__(self, model_path=DEFAULT_PATH):
        self.__initialized = False
        self.model_path = Path(model_path).absolute()
        self.queue = QueueService()

    def _start_consuming(self):
        logging.info('[YoloLegoDetector] Started consuming')
        self.queue.channel.start_consuming()

    def __initialize__(self):
        if self.__initialized:
            raise Exception("YoloLegoDetector already initialized")

        if not self.model_path.exists():
            message = f"[YoloLegoDetector] No model found in {str(self.model_path)}"
            logging.error(message)
            raise RuntimeError(message)

        start_time = time.time()
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=str(self.model_path))
        if torch.cuda.is_available():
            self.model.cuda()

        self.queue.subscribe('detect', self._detect_handler)
        elapsed_time = time.time() - start_time

        t = threading.Thread(target=self._start_consuming, args=())
        t.start()

        logging.info("Loading model took {} seconds".format(elapsed_time))
        self.__initialized = True


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
        logging.info('_detect_handler')
        image = numpy.load(BytesIO(body), allow_pickle=True)

        results = self.detect_lego(image)
        channel.basic_publish(
                exchange='',
                routing_key=properties.reply_to,
                body=results
            )

    def detect_lego(self, image: numpy.ndarray) -> DetectionResults:
        if not self.__initialized:
            logging.info(
                "YoloLegoDetector is not initialized, this process can take a few seconds for the first time.")
            self.__initialize__()

        logging.info("[YoloLegoDetector][detect_lego] Detecting bricks...")
        start_time = time.time()
        results = self.model([image], size=image.shape[0])
        elapsed_time = 1000 * (time.time() - start_time)
        logging.info(
            f"[YoloLegoDetector][detect_lego] Detecting bricks took {elapsed_time} milliseconds")

        return self.convert_results_to_common_format(results)
