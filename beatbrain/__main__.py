from . import cli
from loguru import logger


@logger.catch
def main():
    cli.main()


if __name__ == "__main__":
    main()
