'''
@author: xuxiangfeng 
@date: 2020/1/26
'''

from datetime import datetime, timedelta

# str转换为datetime
dt = datetime.strptime("2020-01-20 12:24:56", "%Y-%m-%d %H:%M:%S")
print(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

# datetime转换为str
now = datetime.now()
print(now.strftime('%Y-%m-%d %H:%M:%S'))

# 把datetime转换为timestamp
dt2 = datetime(2015, 4, 19, 12, 20, 34)  # 用指定日期时间创建datetime
print(dt2.timestamp())

# timestamp转换为datetime
dt3 = datetime.fromtimestamp(1429417200.0)
print(dt3)

# datetime加减
dt4 = now + timedelta(days=2, hours=12)
print(dt4)
