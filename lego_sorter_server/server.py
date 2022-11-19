import grpc

from concurrent import futures
from lego_sorter_server.generated import LegoSorter_pb2_grpc, LegoCapture_pb2_grpc, LegoAnalysis_pb2_grpc
from lego_sorter_server.service.LegoCaptureService import LegoCaptureService
from service.LegoAnalysisService import LegoAnalysisService
from service.LegoSorterService import LegoSorterService

from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig


class Server:

    @staticmethod
    async def run(sorterConfig: BrickCategoryConfig):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=16), options=options)
        l = LegoSorterService(sorterConfig)
        LegoSorter_pb2_grpc.add_LegoSorterServicer_to_server(l, server)
        # LegoCapture_pb2_grpc.add_LegoCaptureServicer_to_server(LegoCaptureService(), server)
        # LegoAnalysis_pb2_grpc.add_LegoAnalysisServicer_to_server(LegoAnalysisService(), server)
        server.add_insecure_port('[::]:50051')
        await server.start()
        await server.wait_for_termination()
