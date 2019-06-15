import fcntl
import tools


import os




class FileLocker:
    def __init__(self, filename):
        self.__filename = filename
        pass

    def acquire(self):
        self.file =  open(self.__filename, 'w+')
        tools.print_refresh(f'acquire {self.__filename}')
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        tools.print_refresh('')

    def release(self):
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()

    def __enter__(self):
        # print(f'acquire file locker {self.__filename}')
        self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print(f'release file locker {self.__filename}')
        self.release()

if __name__ == '__main__':
    import tools
    import time
    with FileLocker('t/a.locker'):
        with open('t/a.txt','a') as f:
            f.write(tools.time_now_str()+'\n')
        time.sleep(5)

    exit()
