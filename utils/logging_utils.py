import logging


def configure_logging(level: int = logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
