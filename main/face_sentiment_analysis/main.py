
import logging


import sys
sys.path.append("..")
from libs import log

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Main.")
    print("Hello")


if __name__=="__main__":

    logger.setLevel(logging.INFO)
    log.add_stream_handler(logger)

    try:
        main()

    except Exception as error:
        logger.exception("Unhandled exception:")
        raise error
