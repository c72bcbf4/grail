import sys

import numpy as np
from loguru import logger

_initialized = False


def configure_logging(filename=None, rank=0, level="INFO"):
    global _initialized

    if _initialized:
        return

    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> {elapsed} <level>{level: <4}</level> "

    color = ",".join(map(str, np.random.randint(50, 200, 3).tolist()))

    fmt_chunks = [
        " <level>",
        f"<fg {color}>",
        "rank {extra[rank]}",
        f"</fg {color}>",
        "</level>",
    ]
    fmt += "".join(fmt_chunks)
    fmt += " {message}"
    args = {"format": fmt, "level": level}
    logger.remove(0)
    logger.add(sys.stdout, **args)

    if filename is not None:
        logger.add(filename, **args)

    if rank is not None:
        logger.configure(extra={"rank": rank})

    _initialized = True
