'''
@author: xuxiangfeng 
@date: 2020/1/4
'''
import baostock as bs
import pandas as pd


class Stock:
    '''
    http://baostock.com/baostock/index.php/
    '''
    _s_minute = "date,time,code,open,high,low,close,volume,amount,adjustflag"
    _s_day = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    _s_long = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
    _dic = {
        '5': _s_minute, '15': _s_minute, '30': _s_minute,
        '60': _s_minute, 'd': _s_day, 'w': _s_long, 'm': _s_long
    }

    @staticmethod
    def query(gid, start_date, end_date, frequency='5'):
        lg = bs.login()
        rs = bs.query_history_k_data_plus(
            gid,
            Stock._dic[frequency],
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="3"
        )
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        result.to_csv("./data/history_stock_%s.csv" % gid, index=False)
        bs.logout()


if __name__ == '__main__':
    Stock.query('sz.000651', '2017-10-10', '2019-10-17')
