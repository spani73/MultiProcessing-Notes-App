import multiprocessing
import logging
from multiprocessing.context import Process
import time
working = None
def work(msg , max):
    name = multiprocessing.current_process().name
    logging.info(f'{name}Started')
    for x in range(max):
        logging.info(f'{name}{msg}')
        time.sleep(1)


def main():
    logging.info('Started')
    max = 2
    worker = Process(target=work, args= ['Working',max],daemon=True,name='Worker')
    worker.start()
    working = worker
    
    time.sleep(5)

    if worker.is_alive():
        worker.terminate()

    
    logging.info(f'Finished : {worker.exitcode}')






logging.basicConfig(format= '%(levelname)s- %(asctime)s: %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)

if __name__ == "__main__":
    main()
