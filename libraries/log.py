import logging 

logging.basicConfig(
    format='%(asctime)s | %(name)s >> %(filename)10s:%(lineno)03d %(levelname)7s %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger('GPT-SWARM')

if __name__ == '__main__':
    logger.debug(' ... log was initialized ... ')