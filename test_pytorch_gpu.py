import datetime
import logging
import os

from logging import Logger
import torch

class ArgsObject:
    def __init__(
        self, 
        logger_name: str = "default",
        logger_dir: str = "logs",
    ) -> None:
        self.logger_name = logger_name
        self.logger_dir = logger_dir
        

def set_logger(args: ArgsObject) -> Logger:
    if not os.path.exists(args.logger_dir):
        os.makedirs(args.logger_dir)

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name = f"{args.logger_name}_{time}.log"

    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.logger_dir, file_name))
    fh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(name)s - %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    args = ArgsObject(logger_name="default", logger_dir="logs")
    logger = set_logger(args)

    logger.info(f"Torch version: {torch.__version__}")
    logger.info(f"Torch cuda available: {torch.cuda.is_available()}")