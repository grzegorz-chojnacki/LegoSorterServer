import logging
from typing import Tuple, List

import numpy
import pickle
import asyncio

from PIL.Image import Image

from lego_sorter_server.analysis.classification.ClassificationResults import ClassificationResults
from lego_sorter_server.analysis.classification.LegoClassifierProvider import LegoClassifierProvider
from lego_sorter_server.analysis.classification.classifiers.TFLegoClassifier import TFLegoClassifier
from lego_sorter_server.analysis.detection import DetectionUtils
from lego_sorter_server.analysis.detection.DetectionResults import DetectionResults
from lego_sorter_server.analysis.detection.detectors.LegoDetector import LegoDetector
from lego_sorter_server.analysis.detection.detectors.LegoDetectorProvider import LegoDetectorProvider

from lego_sorter_server.service.QueueService import QueueService
from io import BytesIO
import multiprocessing as mp

class AnalysisService:
    DEFAULT_IMAGE_DETECTION_SIZE = (640, 640)
    BORDER_MARGIN_RELATIVE = 0.001

    def __init__(self):
        self.queue = QueueService()
        self.queue.startInThread()
        self.detectorPool = [self.startDetector(i) for i in range(1)]
        self.classifierPool = [self.startClassifier(i) for i in range(1)]

    def startDetector(self, i):
        p = mp.Process(
            name=f'Detector-{i:02}',
            target=LegoDetectorProvider.get_default_detector)
        p.start()
        return p

    def startClassifier(self, i):
        p = mp.Process(
            name=f'Classifier-{i:02}',
            target=LegoClassifierProvider.get_default_classifier)
        p.start()
        return p


    def _rescale(self, image: Image, resize: bool, discard_border_results: bool):
        scale = 1
        original_size = image.size

        if image.size != self.DEFAULT_IMAGE_DETECTION_SIZE:
            if resize is False:
                logging.warning(f"[AnalysisService] Requested detection on an image with a non-standard size "
                                f"{image.size} but 'resize' parameter is {resize}.")
            else:
                logging.info(f"[AnalysisService] Resizing an image from "
                             f"{image.size} to {self.DEFAULT_IMAGE_DETECTION_SIZE}")
                image, scale = DetectionUtils.resize(
                    image, self.DEFAULT_IMAGE_DETECTION_SIZE[0])

        accepted_xy_range = [
            (original_size[0] * scale) / image.size[0],
            (original_size[1] * scale) / image.size[1]
        ] if discard_border_results else [1, 1]

        return image, scale, original_size, accepted_xy_range

    async def detect(self, image: Image, resize: bool = True, threshold=0.5,
                     discard_border_results: bool = True) -> DetectionResults:
        image, scale, original_size, accepted_xy_range = self._rescale(
            image, resize, discard_border_results)

        body = BytesIO()
        numpy.save(body, numpy.array(image), allow_pickle=True)

        detection_results = pickle.loads(await self.queue.rpc('detect', body.getvalue()))
        detection_results = self.filter_detection_results(
            detection_results, threshold, accepted_xy_range)

        return self.translate_bounding_boxes_to_original_size(
            detection_results,
            scale,
            original_size,
            self.DEFAULT_IMAGE_DETECTION_SIZE[0])

    async def classify(self, images: List[Image]) -> ClassificationResults:
        async def handle(image):
            body = BytesIO()
            numpy.save(body, numpy.array(image), allow_pickle=True)
            return pickle.loads(await self.queue.rpc('classify', body.getvalue()))

        results = await asyncio.gather(*[handle(image) for image in images])

        return ClassificationResults(
            [r.classification_classes[0] for r in results],
            [r.classification_scores[0] for r in results],
        )

    def _crop_bricks(self, image: Image, detection_results: List[Image]) -> List[Image]:
        return [
            DetectionUtils.crop_with_margin_from_bb(image, bounding_box)
            for bounding_box in detection_results.detection_boxes
        ]

    async def detect_and_classify(self, image: Image,
                                  detection_threshold: float = 0.5,
                                  discard_border_results: bool = True) \
            -> Tuple[DetectionResults, ClassificationResults]:

        detection_results = await self.detect(
            image,
            threshold=detection_threshold,
            discard_border_results=discard_border_results)

        brick_images = self._crop_bricks(image, detection_results)
        classification_results = await self.classify(brick_images)

        return detection_results, classification_results

    @staticmethod
    def translate_bounding_boxes_to_original_size(
            detection_results: DetectionResults,
            scale: float,
            target_image_size: Tuple[int, int],
            detection_image_size: int = 640) -> DetectionResults:

        bbs = []
        for i in range(len(detection_results.detection_classes)):
            y_min, x_min, y_max, x_max = [
                int(coord * detection_image_size * 1 / scale)
                for coord in detection_results.detection_boxes[i]
            ]

            # if y_max >= target_image_size[1] or x_max >= target_image_size[0]:
            #     continue
            y_max = min(y_max, target_image_size[1])
            x_max = min(x_max, target_image_size[0])

            bbs.append((y_min, x_min, y_max, x_max))

        detection_results_translated = DetectionResults(
            detection_results.detection_scores,
            detection_results.detection_classes,
            bbs)

        return detection_results_translated

    def filter_detection_results(self, detection_results, threshold, accepted_xy_range):
        limit = len(detection_results.detection_scores)
        for index, score in enumerate(detection_results.detection_scores):
            if score < threshold:
                limit = index
                break

        filtered_results = DetectionResults(detection_results.detection_scores[:limit],
                                            detection_results.detection_classes[:limit],
                                            detection_results.detection_boxes[:limit])
        if accepted_xy_range != [1, 1]:
            results = []
            for score, clazz, box in zip(filtered_results.detection_scores,
                                         filtered_results.detection_classes,
                                         detection_results.detection_boxes):
                # (ymin, xmin, ymax, xmax)
                if box[0] < self.BORDER_MARGIN_RELATIVE \
                        or box[1] < self.BORDER_MARGIN_RELATIVE \
                        or box[2] > accepted_xy_range[1] - self.BORDER_MARGIN_RELATIVE \
                        or box[3] > accepted_xy_range[0] - self.BORDER_MARGIN_RELATIVE:
                    continue
                results.append([score, clazz, box])

            if len(results) != 0:
                results = numpy.stack(results)
                filtered_results = DetectionResults(
                    results[:, 0], results[:, 1], results[:, 2])
            else:
                filtered_results = DetectionResults([], [], [])

        return filtered_results
