import argparse

from lego_sorter_server.server import Server
import logging
import sys
import threading
import asyncio

from lego_sorter_server.service.BrickCategoryConfig import BrickCategoryConfig


def exception_handler(exc_type, value, tb):
    logging.exception(f"Uncaught exception: {str(value)}")


def configureLogging(logger):
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(levelname)s][%(processName)s]%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--brick_category_config", "-c", help='.json file with brick-category mapping specification', type=str, required=False)
    args = parser.parse_args()

    configureLogging(logging.getLogger())

    sys.excepthook = exception_handler
    threading.excepthook = exception_handler

    asyncio.run(Server.run(BrickCategoryConfig(args.brick_category_config)))
