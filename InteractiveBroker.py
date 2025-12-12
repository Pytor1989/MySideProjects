import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ibapi.client import *
from ibapi.wrapper import *
from ibapi.ticktype import TickTypeEnum
import datetime
import time
import threading
import mySQLhandler #mySQLhandler can be found under my other project SqlManip.py

class InstrChar(EClient, EWrapper):
    def __init__(self):
        self.dfChar = pd.DataFrame(columns=['CONID','SYMBOL','SEC_TYPE','EXPIRY_DATE','LAST_TD_DATE','CLIENT_ID','UNK1','MULTI',
                                            'EXCHANGE','PRIM_EXCHANGE','CCY','TICKER','SHORT_TICKER','COND','UNK3','UNK4','UNK5','EPT',
                                            'NAME','SECTOR','INDUSTRY','SUBINDUSTRY','STOCKTYPE'])
                                            # Dont change those column name or do it after checking contractDetails() function ouput and 
                                            # tweak it using the attr to see all the attributes and order them
      EClient.__init__(self, self)

    def nextValidId(self, orderId):
        self.orderId = orderId

    def nextId(self):
        self.orderId += 1
        return self.orderId

    def error(self, reqId, errorCode, errorString, advancedOrderReject):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")


    def contractDetails(self, reqId, contractDetails):
        attrs = vars(contractDetails)
        list_attrib = list(str(attrs['contract']).split(',')) #list(filter(None,list(str(attrs['contract']).split(','))))
        list_attrib.extend([attrs['longName'],attrs['industry'],attrs['category'],attrs['subcategory'], attrs['stockType']])
        self.dfChar.loc[len(self.dfChar)] = list_attrib


    def contractDetailsEnd(self, reqId):
        print(
            datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "contractDetailsEnd.",
            f"reqId:{reqId}",
        )
        self.disconnect()
        print('Request Disconnected')



class InstrBarDaily(EClient, EWrapper):
    def __init__(self):
        self.dfBar = pd.DataFrame(columns=['DATE','OPEN','HIGH','LOW','CLOSE','VOLUME','CONID','SECTYPE'])
        self.histoTime = None
        EClient.__init__(self, self)

    def nextValidId(self, orderId):
        self.orderId = orderId

    def nextId(self):
        self.orderId += 1
        return self.orderId

    def error(self, reqId, errorCode, errorString, advancedOrderReject):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")


    def headTimestamp(self, reqId, headTimeStamp):
        '''See how far in the past i can request Histo data'''
        self.histoTime = datetime.datetime.fromtimestamp(int(headTimeStamp)).date()
        self.cancelHeadTimeStamp(reqId)
        self.disconnect()

    # Histo Data:
    def historicalData(self, reqId, bar):
        # inDate = bar.date[:17]
        # d = datetime.datetime.strptime(inDate, "%Y%m%d %H:%M:%S")
        # listVal = [d, bar.open, bar.high, bar.low, bar.close,float(bar.volume)]
        self.dfBar.loc[len(self.dfBar)] = [datetime.datetime.strptime(bar.date[:8], "%Y%m%d"), bar.open, bar.high, bar.low, bar.close,float(bar.volume),None,None]

    def historicalDataEnd(self, reqId, start, end):
        # print(f"Historical Data Ended for {reqId}. Started at {start}, ending at {end}")
        self.cancelHistoricalData(reqId)
        self.disconnect()
        print('Request Disconnected')







# InstrChar is used to fetch instruments attributes (id, ticker, various specific information)
# InstrBarDaily is used to fetch daily data for all asset classes.
###### Example: (after being connected to your IB platform)
def instrChar(yourConnectionInfoasList,_conId):
    app = InstrChar()
    app.connect(yourConnectionInfoasList[0],yourConnectionInfoasList[1],yourConnectionInfoasList[2])
    threading.Thread(target=app.run).start()
    time.sleep(2)
    print('Request Connected')
    app.reqContractDetails(app.nextId(), contract.conId =_conId)
    time.sleep(1)
    # send it to sql or ouput it using app.dfChar to fetch the dataframe with the data

# using this function you can fetch the instrument (on by one, you should loop it to for a list of conid) characteristics by inputing the conid of the instrument
# for futures you should also put the SECTYPE = FUT, or CONFUT for continuous future as they share the same conid
#for prices:
def instrDaily(yourConnectionInfoasList,_conId,_duration,wtc='TRADES'):
    '''duration param should be 'x D/W/M/Y' '''
    # modifier plus tard quand tqble cree dans SQL
    app = InstrBarDaily()
    app.connect(yourConnectionInfoasList[0],yourConnectionInfoasList[1],yourConnectionInfoasList[2])
    threading.Thread(target=app.run).start()
    time.sleep(2)
    print('Request Connected')
    app.reqHistoricalData(app.nextId(),  contract.conId =_conId, "", _duration, "1 day", wtc, 0, 1, False, [])
    time.sleep(3)
    df = app.dfBar
    time.sleep(5)
    print(df.tail(2))


# for CASH instruments wtc = MIDPOINT, fetch the price serie OHLC for IB conid using the _duration param = 1 W for one week, 3 M for 3 month etc
# you can tweak the reqHistoricalData to get intraday or option chain using the videos on the ibapi tutorials
