import datetime
import logging
import os

from logging import Logger
import tensorflow as tf


def set_tensorflow_config():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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
    # Make sure that tensorflow doesn't take away all memory at initialization. 
    set_tensorflow_config()
    
    args = ArgsObject(logger_name="default", logger_dir="logs")
    logger = set_logger(args)

    logger.info(f"Tensorflow version: {tf.__version__}")
    logger.info(f"Tensorflow cuda available: {tf.test.is_gpu_available()}")