'''
@author: xuxiangfeng 
@date: 2020/1/2
'''
import logging


class MyLogger:

    @staticmethod
    def get_log(log_file_name: str, log_level: str, logger_name: str = None):
        '''
        log_file_name: 日志文件的路径，例: ../code/mylog.txt
        log_level: 日志记录的等级，['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
        '''
        # 创建一个logger
        logger = logging.getLogger(logger_name)

        # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(log_level)

        # 创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file_name)

        # 创建一个handler用于输出控制台
        # console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line: %(lineno)d]-%(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        # 给logger添加handler
        logger.addHandler(file_handler)
        # self.__logger.addHandler(console_handler)

        return logger

if __name__ == '__main__':
    log = MyLogger.get_log(log_file_name='log.txt', log_level="INFO")
    log.debug('hhhhh ... ')
    log.info('hello ... ')
