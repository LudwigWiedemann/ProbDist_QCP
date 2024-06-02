import logging
import time
import save

logger = logging


class Logger(object):
    logger.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    filename = save.start_time+"-Logger"
    logFormatter = logger.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logger.getLogger()
    fileHandler = logger.FileHandler("{0}/{1}.log".format("..\Logger", filename))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logger.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
