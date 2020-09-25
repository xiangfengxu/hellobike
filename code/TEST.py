'''
@author: xuxiangfeng 
@date: 2020/1/3
'''

import pandas as pd
import redis
import json
from datetime import datetime

r = redis.Redis(host='switch-bos.redis.ttbike.com.cn',
                    port=8773,
                    db=1,
                    password='hRVwAafjTXyCchRVw')


def get_site_info(site_guid, dt):
    key = "algRcd:%s_%s" % (site_guid, dt)
    res = [0,0,0,0,0]
    if r.get(key):
        content = json.loads(r.get(key))
        for bike in content['bikelist']:
            if bike['percent'] < 20:
                res[0] += 1
            elif bike['percent'] < 40:
                res[1] += 1
            elif bike['percent'] < 60:
                res[2] += 1
            elif bike['percent'] < 80:
                res[3] += 1
            else:
                res[4] += 1
    return res

now = datetime.now()
dt = datetime(now.year, now.month, now.day, now.hour, 0).strftime("%Y%m%d%H%M")
df = pd.read_csv("/workspace/carkey/pythonWorkDir/xuxiangfeng/temp/ev_bike_parking_ruian.csv")

df[dt] = df.apply(lambda row: get_site_info(row['site_guid'], dt), axis=1)


df.to_csv('/workspace/carkey/pythonWorkDir/xuxiangfeng/temp/ev_bike_parking_ruian.csv', header=True, index=False)


