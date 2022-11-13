from lego_sorter_server.analysis.AnalysisService import AnalysisService
import logging
import time

from PIL import Image
from typing import TypeAlias, Callable
from lego_sorter_server.generated import LegoAnalysis_pb2_grpc
from lego_sorter_server.generated.Messages_pb2 import ImageRequest, ListOfBoundingBoxes
from lego_sorter_server.service.ImageProtoUtils import ImageProtoUtils

AnalysisFn: TypeAlias = Callable[[Image.Image], ListOfBoundingBoxes]

class LegoAnalysisService(LegoAnalysis_pb2_grpc.LegoAnalysisServicer):
    def __init__(self):
        self.analysis_service = AnalysisService()

    def DetectBricks(self, request: ImageRequest, context):
        return self._process_bricks(request, 'Detecting and preparing', self._detect)

    def DetectAndClassifyBricks(self, request: ImageRequest, context):
        return self._process_bricks(request, 'Detecting, classifying and preparing', self._detect_classify)

    def _detect(self, image: Image.Image) -> ListOfBoundingBoxes:
        detection_results = self.analysis_service.detect(image)
        return ImageProtoUtils.prepare_bbs_response_from_detection_results(detection_results)

    def _detect_classify(self, image: Image.Image) -> ListOfBoundingBoxes:
        detection_results, classification_results = self.analysis_service.detect_and_classify(
            image)
        return ImageProtoUtils.prepare_response_from_analysis_results(detection_results, classification_results)

    def _process_bricks(self, request: ImageRequest, handler_name: str, handler_fn: AnalysisFn) -> ListOfBoundingBoxes:
        logging.info("[LegoAnalysisService] Request received, processing...")
        start_time = time.time()

        image = ImageProtoUtils.prepare_image(request)
        results = handler_fn(image)

        elapsed_millis = int((time.time() - start_time) * 1000)
        logging.info(
            f"[LegoAnalysisService] f{handler_name} response took {elapsed_millis} milliseconds.")

        return results
