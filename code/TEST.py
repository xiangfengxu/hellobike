'''
@author: xuxiangfeng 
@date: 2020/1/3
'''


from mylogger import MyLogger
import logging


# log1 = MyLogger.get_log(log_file_name='log.txt', log_level="DEBUG")
# log2 = MyLogger.get_log(log_file_name='log.txt', log_level="DEBUG")

log1 = logging.getLogger()
log2 = logging.getLogger('frank')
print(id(log1))
print(id(log2))

# try:
#     1 / 0
# except:
#     log.info("0不能做除数,hhh")
#     log.debug("hello jack")
