import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import myLib
from dateutil.relativedelta import relativedelta
import os
import sys
import contextlib
import io
import mySQLhandler


# period = ['1mo', '1y', '5y','max']
# interval = ['1d','1wk','1mo']


class InstrData(object):
    def __init__(self, _ticker):
        self._ticker_str = _ticker
        self._ticker = yf.Ticker(_ticker)

    def is_valid(self):
        """Check validity without noisy HTTP error output."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            data = self._ticker.history(period="1d")
        return not data.empty

    def ticker_(self):
        '''getter'''
        return self._ticker

    def get_info(self):
        return self._ticker.info

    def get_floatInfo(self,info_):
        # return float(self._ticker.info[info_])
        data = self._ticker.info.get(info_)
        val = float(data if data is not None else 1)
        return val

    def get_dtInfo(self,info_):
        return datetime.datetime.fromtimestamp(int(self._ticker.info[info_]))

    def price(self,period_='1y'):
        df = self._ticker.history(period=period_).reset_index()
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.strftime('%Y-%m-%d')))
        return df

    def cust_price(self,start_='1989-04-10', end_=datetime.datetime.today().strftime('%Y-%m-%d'), interval_='1d'):
        df = self._ticker.history(start=start_, end=end_, interval=interval_).reset_index()
        df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.strftime('%Y-%m-%d')))
        return df

    def get_price_OHLC(self,period_='1y'):
        df = self.price(period_)
        df = df[['Date','Open','High','Low','Close','Volume']]
        df['Volume'] = df['Volume'].astype(float)
        return df

    def get_price_close(self, period_='1y'):
        df = self.price(period_)
        df = df[['Date','Close','Volume']]
        return df

    def get_last_close(self,dtEnd):
        startDt = str(pd.Timestamp(dtEnd) - relativedelta(days=10))[:10]
        df = self.get_cust_price_close(startDt, dtEnd)
        return df[df['Date'] <= dtEnd].tail(1)['Close'].squeeze()

    def get_cust_price_OHLC(self, start_='1989-04-10', end_=datetime.datetime.today().strftime('%Y-%m-%d'), interval_='1d'):
        df = self.cust_price(start_, end_, interval_)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Volume'] = df['Volume'].astype(float)
        return df

    def get_cust_price_close(self, start_='1989-04-10', end_=datetime.datetime.today().strftime('%Y-%m-%d'), interval_='1d'):
        df = self.cust_price(start_, end_, interval_)
        df = df[['Date','Close','Volume']]
        return df



class Stocks(InstrData):
    def __init__(self, _ticker):
        # call parent constructor so InstrData is initialized
        super().__init__(_ticker)

    def get_stock_isin(self):
        return self._ticker.get_isin()


    def get_dvd(self, period_='1y'):
        df = self.price(period_)
        df = df[['Date', 'Dividends']]
        df = df[df['Dividends'] != 0.00]
        return df

    def get_cust_dvd(self, start_='1989-04-10', end_=datetime.datetime.today().strftime('%Y-%m-%d'), interval_='1d'):
        df = self.cust_price(start_, end_, interval_)
        df = df[['Date', 'Dividends']]
        df = df[df['Dividends'] != 0.00]
        return df

    def get_stocksplit(self, period_='1y'):
        df = self.price(period_)
        df = df[['Date', 'Stock Splits']]
        df = df[df['Stock Splits'] != 0.00]
        return df

    def get_cust_stocksplit(self, start_='1989-04-10', end_=datetime.datetime.today().strftime('%Y-%m-%d'),
                            interval_='1d'):
        df = self.cust_price(start_, end_, interval_)
        df = df[['Date', 'Stock Splits']]
        df = df[df['Stock Splits'] != 0.00]
        return df

    def get_nb_shares(self, date_):
        '''Get nber of shares outstanding at a certain date YYYY-MM-DD'''
        df = self._ticker.get_shares_full().reset_index()
        df['Date'] = pd.to_datetime(df['index'].apply(lambda x: x.strftime('%Y-%m-%d')))
        df = df[df['Date'] <= date_].tail(1)[['Date', 0]]
        list_col_shares = ['Date', 'Nb_Shares']
        df = myLib.rename_cols(list_col_shares, df)
        return df

    def get_calend_raw(self):
        dict_cal = self._ticker.get_calendar()
        return dict_cal

    def get_div_cal(self):
        df = pd.DataFrame([self.get_calend_raw()])[['Dividend Date','Ex-Dividend Date']]
        col_dt_cal = ['Dividend Date', 'Ex-Dividend Date']
        df = myLib.apply_dates(col_dt_cal, df, format_='%Y-%m-%d', yearfirst_=True)
        return df

    def get_earnings_dt(self):
        earning_dict = self.get_calend_raw()
        period_dt = (earning_dict.get('Earnings Date') or [None])[0]
        return period_dt

    def get_nxt_earnings_est(self):
        df = pd.DataFrame([self.get_calend_raw()])[['Earnings High','Earnings Low','Earnings Average','Revenue High','Revenue Low','Revenue Average']].astype(float)
        return df

    def get_bs_raw(self,freq_='quarterly'):
        '''get balance sheet data with various frequencies without any treatment'''
        df = self._ticker.get_balance_sheet(freq=freq_).transpose().reset_index().rename(columns={'index':'Date'})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def get_is_raw(self,freq_='quarterly'):
        '''get IS data with various frequencies without any treatment'''
        df = self._ticker.get_income_stmt(freq=freq_).transpose().reset_index().rename(columns={'index':'Date'})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def get_cf_raw(self,freq_='quarterly'):
        '''get CF statement data with various frequencies without any treatment'''
        df = self._ticker.get_cashflow(freq=freq_).transpose().reset_index().rename(columns={'index':'Date'})
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def yr_consolidated(self):
        '''Bring together the Balance sheet / Income Statement / CF statement on a yearly basis '''
        freqY = 'yearly'
        lstYR = [self.get_bs_raw(freqY).head(4).reset_index(drop=True).set_index('Date'),
                 self.get_is_raw(freqY).head(4).set_index('Date'),
                 self.get_cf_raw(freqY).head(4).set_index('Date')]
        dfYR = lstYR[0].join([lstYR[1], lstYR[2]], how="inner")  # or "outer"
        dfYR = dfYR.reset_index().rename(columns={'index':'Date'})
        dfYR['Type'] = 'YR'
        return dfYR

    def ltm_consolidated(self):
        '''Bring together the Balance sheet / Income Statement / CF statement on a LTM basis '''
        freqQ = 'quarterly'
        lstLTM = [self.get_bs_raw(freqQ).head(1).reset_index(drop=True),
                  self.get_is_raw(freqQ).drop('Date', axis=1).head(4).sum().to_frame().T,
                  self.get_cf_raw(freqQ).drop('Date', axis=1).head(4).sum().to_frame().T]
        # date = instr.get_dtInfo('mostRecentQuarter').date()
        dfLTM = pd.concat([lstLTM[0], lstLTM[1], lstLTM[2]], axis=1)
        dfLTM['Type'] = 'LTM'
        return dfLTM


    def financials_consolidated(self):
        '''Create a report with yearly and LTM data combined for analysis purpose'''
        df_list = [self.yr_consolidated(), self.ltm_consolidated()]
        cols_fin_list = [x for x in list(df_list[0]) if x in list(df_list[1])]
        list_fin_df = []
        for df in df_list:
            df = df[cols_fin_list]
            list_fin_df.append(df)

        df_meta = pd.concat(list_fin_df).sort_values(['Type', 'Date'], ascending=[True, False]).reset_index(drop=True)
        return df_meta

    def get_fcf_data(self,mode_):
        dataCols = ['TotalLiabilitiesNetMinorityInterest', 'LongTermDebtAndCapitalLeaseObligation', 'CurrentAssets','Inventory',
            'AccountsReceivable','CashCashEquivalentsAndShortTermInvestments', 'FreeCashFlow']
        # dataColsI = ['TotalLiabilitiesNetMinorityInterest', 'LongTermDebtAndCapitalLeaseObligation', 'CurrentAssets',
        #             'AccountsReceivable', 'CashCashEquivalentsAndShortTermInvestments', 'FreeCashFlow']
        if mode_ == 'YR':
            df = self.yr_consolidated().reindex(columns=dataCols)
        else:
            df = self.financials_consolidated().reindex(columns=dataCols)
        return df

    # fcf indic
    def validFCF(self, mode_):
        data = self.get_fcf_data(mode_).head(4)
        data['Increasing'] = data['FreeCashFlow'].diff(-1) > 0
        return  data['Increasing'][:-1].all()

    def nnwc(self,mode_):
        colsNNWC = ['CashCashEquivalentsAndShortTermInvestments', 'AccountsReceivable', 'Inventory',
                    'TotalLiabilitiesNetMinorityInterest']
        nnwcData = self.get_fcf_data(mode_)[colsNNWC].astype(float).fillna(0).head(2)
        nnwcData['NNWC'] = ((nnwcData['CashCashEquivalentsAndShortTermInvestments'] + nnwcData['Inventory'] * .5 +
                             nnwcData['AccountsReceivable'] * .75) - nnwcData['TotalLiabilitiesNetMinorityInterest']).diff(-1) > 0
        return nnwcData['NNWC'][:-1].all()

    def ncav(self,mode_):
        colsNCAV = ['CurrentAssets', 'TotalLiabilitiesNetMinorityInterest']
        ncavData = self.get_fcf_data(mode_)[colsNCAV].astype(float).fillna(0).head(1)
        ncavData['EV'] = self.get_floatInfo('enterpriseValue')
        return ((ncavData['CurrentAssets'] - ncavData['TotalLiabilitiesNetMinorityInterest']) / ncavData['EV']).squeeze()

    def pfcf(self,mode_):
        fcf = self.get_fcf_data(mode_)['FreeCashFlow'].astype(float).fillna(1).head(1).squeeze()
        sharesOut = self.get_floatInfo('sharesOutstanding')
        if mode_ == 'YR':
            dtRpt = str(self.get_dtInfo('lastFiscalYearEnd'))[:10]
        else:
            dtRpt = str(self.get_dtInfo('mostRecentQuarter'))[:10]
        lastClose = self.get_last_close(dtRpt)
        return lastClose/(fcf/sharesOut) # return float

    def fcfDebtR(self,mode_):
        colsFCFD = ['LongTermDebtAndCapitalLeaseObligation','FreeCashFlow']
        fcfD =  self.get_fcf_data(mode_)[colsFCFD].astype(float).fillna(1).head(3)
        fcfD['Ratio'] = fcfD['FreeCashFlow']/fcfD['LongTermDebtAndCapitalLeaseObligation']
        return  (fcfD['Ratio'][:-2] > 0.1).squeeze()

    def fcfDebtI(self,mode_):
        colsFCFD = ['LongTermDebtAndCapitalLeaseObligation','FreeCashFlow']
        fcfD =  self.get_fcf_data(mode_)[colsFCFD].astype(float).fillna(1).head(3)
        fcfD['Ratio'] = fcfD['FreeCashFlow']/fcfD['LongTermDebtAndCapitalLeaseObligation']
        fcfD['Increasing'] = fcfD['Ratio'].diff(-1) > 0
        return  fcfD['Increasing'][:-2].squeeze()


    def evFcf(self,mode_):
        fcf = self.get_fcf_data(mode_)['FreeCashFlow'].astype(float).fillna(1).head(1).squeeze()
        ev = self.get_floatInfo('enterpriseValue')
        return ev/fcf # return float

    def lastFCF(self,mode_):
        return self.get_fcf_data(mode_)['FreeCashFlow'].astype(float).fillna(0).head(1).squeeze()


def importInstr(_connListSQL, tickList):
    lenUpload, listInstr, countImp = len(tickList), [], 0
    sqlManip = mySQLhandler.SqlManip(_connListSQL)
    sqlManip.deleteSimple('assetNewYF', 'staging')
    for ticker in tqdm(tickList, total=len(tickList), desc='Processing tickers creation'):
        instr = InstrData(ticker)
        info = instr.get_info()
        if "longName" not in info:
            info["longName"] = info.get("shortName")
        if info.get("typeDisp") == "Equity" and "longBusinessSummary" not in info and "bookValue" not in info:
            info["typeDisp"] = "ETF"
        row = {
            "longName": info.get("longName"),
            "typeDisp": info.get("typeDisp"),
            "currency": info.get("currency"),
            "fullExchangeName": info.get("fullExchangeName"),
            "sector": info.get("sector", ""),  # ETFs don't have sector
            "industry": info.get("industry", ""),  # ETFs don't have industry
            "SYMBOL": ticker
        }
        listInstr.append(row)
        countImp += 1
    if listInstr:
        print(f"Preparing sql import for {countImp}/{lenUpload} instrument requested ")
        dfInstr = pd.DataFrame(listInstr)
        sqlManip.importData(dfInstr, 'assetNewYF', 'staging')
        # add stored procedure
        sp = '[dbo].[sp_NewAssetYF]'
        sqlManip.cursor(sp)
        print('Instruments imported in relevant table')


def requestHistStampBulk(_connListSQL,listYF,_duration):
    dateStart, listDt, countImport,listValidCont =  myLib.offsetDate(_duration), [], 0,[]
    if listYF is None:
        pass
    else:
        lenUpload = len(listYF)
        for i in listYF:
            ticker = i[0]
            instr = InstrData(ticker)
            ms = instr.get_info()['firstTradeDateMilliseconds']
            if ms < 0:
                instStartDt = datetime.date(1971, 1, 1)
            else:
                instStartDt = datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).date()

            listDt.append([i[0], dateStart, instStartDt, i[6],i[1],i[7]])
            countImport += 1
            print(f"Requested {countImport}/{lenUpload}: {i[6]} into List")

        for d in listDt:
            if d[1] > d[2]:
                print(f"=> Request price history for {_duration} ok for {d[3]} | {d[0]}")
                listValidCont.append([d[0], d[1], d[3],d[4],d[5]])
            else:
                print(f"=>ERROR: Requested to much history: minimum {str(d[2])} for {d[3]} | {d[0]}")
        return listValidCont


def requestHistoPriceDailyBulk(_connListSQL,_listValidCont,_duration):
    if not _listValidCont:
        print('No instruments available for the selected date/class/strategy')
    else:
        sqlManip = mySQLhandler.SqlManip(_connListSQL)
        lenUpload,countInstr,listPx, sourceId = len(_listValidCont),0,[],2
        for v in _listValidCont:
            countInstr += 1
            instr = InstrData(v[0])
            df =  instr.get_cust_price_OHLC(str(v[1]))
            df['CONID'],df['SECTYPE'] = v[4],v[3]
            listPx.append(df)
            print(f"Uploaded {countInstr}/{lenUpload}: {v[2]} | {v[0]} in staging Table")
        countYFimp = len(listPx)
        dfPrice = pd.concat(listPx)
        if countYFimp == lenUpload:
            sqlManip.deleteSimple('priceDaily', 'staging')
            sqlManip.importData(dfPrice,'priceDaily', 'staging')
            spPrice = '[dbo].[sp_InsertPriceDaily]'+str(sourceId)
            sqlManip.cursor(spPrice)
            print('=> Instrument price daily imported in relevant table')
        else:
            print('Import Fail! Check logs and retry')



def requestHistoPriceDailyMax(_connListSQL,_listValidCont):
    listPrice, sourceYF = [], 2
    sqlManip = mySQLhandler.SqlManip(_connListSQL)
    for i in _listValidCont:
        instr = InstrData(i[0])
        ms = instr.get_info()['firstTradeDateMilliseconds']
        if ms < 0:
            dtStart = str(datetime.date(1971, 1, 1))
        else:
            dtStart = str(datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).date())
        data = instr.get_cust_price_OHLC(dtStart)
        data['CONID'], data['SECTYPE'] = i[7], i[1]
        listPrice.append(data)
        print(f"Price data requested for {i[6]} | {i[0]} since {dtStart}")
    df = pd.concat(listPrice)
    sqlManip.deleteSimple('priceDaily', 'staging')
    sqlManip.importData(df, 'priceDaily', 'staging')
    spPrice = '[dbo].[sp_InsertPriceDaily]' + str(sourceYF)
    sqlManip.cursor(spPrice)
    print('=> Instrument price daily imported in relevant table')

def get_validTickersYF(dictTick):
    '''Return a dict of Valid and Invalid tickers for Yahoo'''
    dictValid, dictInvalid = {}, {}
    count, dataset = 0, len(dictTick)
    for key, value in tqdm(dictTick.items(), total=len(dictTick), desc="Processing tickers"):
        instr = InstrData(key)
        if instr.is_valid():
            dictValid[key] = value
            count += 1
        else:
            dictInvalid[key] = value
    print(f"{count}/{dataset} valid ticker for yF")
    return dictValid, dictInvalid

def validCondFCF(lstConst, mode_):
    dictSelected = {}
    countSel, validImp = 0, len(lstConst)
    for lstValue in tqdm(lstConst, total=len(lstConst), desc="Processing selection by setting FCF parameter"):
        instr = Stocks(lstValue[0])
        if instr.validFCF(mode_):
            countSel += 1
            cond = 1
        else:
            cond=0
        dictSelected[lstValue[0]] = {"NAME": lstValue[6], "CONID": lstValue[7],"EQINDEX_ISO": lstValue[4], "CONDITION": cond}

    print(f"{countSel}/{validImp} passed the first selection filter")
    df = pd.DataFrame.from_dict(dictSelected, orient="index")

    return df

def getEarningsDtConst(lstConst_):
    dictE = {}
    for const in tqdm(lstConst_, total = len(lstConst_), desc="Processing Earnings Date for Constituents"):
        instr = Stocks(const[0])
        lastE = instr.get_dtInfo('mostRecentQuarter')
        nxtFiscE = instr.get_dtInfo('nextFiscalYearEnd')
        nxtEannounce = instr.get_earnings_dt()
        dictE[const[0]] = {"CONID": const[7], "EQINDEX_ISO": const[4], "LAST_REPORT_DATE": lastE,
                           "NXT_FISC_YE_DATE": nxtFiscE, 'NXT_ANN_DATE': nxtEannounce}

    df = pd.DataFrame.from_dict(dictE, orient="index")
    return df

def snapFCF(lstConst_, mode_):
    dictResult = {}
    for cont in tqdm(lstConst_, total=len(lstConst_), desc="Processing selection, delivering final report"):
        data = Stocks(cont[0])
        dictVal = {'CONID': cont[7], 'EQINDEX_ID': cont[5], 'NNWC': data.nnwc(mode_), 'NCAV': data.ncav(mode_),
                   'PFCF': data.pfcf(mode_), 'FCFDEBT_R': data.fcfDebtR(mode_),
                   'FCFDEBT': data.fcfDebtI(mode_), 'EVFCF': data.evFcf(mode_), 'LAST_FCF': data.lastFCF(mode_),
                   'LAST_NBSHARES': data.get_floatInfo('sharesOutstanding'), 'LAST_ANN_DT':  cont[3]}
        dictResult[cont[0]] = dictVal

    df = pd.DataFrame.from_dict(dictResult, orient='index').reset_index().rename(columns={'index': 'SYMBOL'})
    myLib.to_excel_colsize_multitab('SNAP_FCF', [df], ['FCF_Results'], myLib.pthDbReport)
    print(f"Snap Age of Empire Report exported for selected index to IB|DataBase|Import\nAnalyze it before running Import File and Run insert FCF")


def techFCF(tickerDict):
    dataWatch = []
    for ticker in tickerDict.keys():
        instr = InstrData(ticker)
        data = instr.get_cust_price_OHLC(myLib.offsetDate('5 Y'))
        data = data.sort_values('Date').reset_index(drop=True)
        data["EMA350"] = data["Close"].ewm(span=350, adjust=False).mean()
        data["EMA25"] = data["Close"].ewm(span=25, adjust=False).mean()
        ema350_last = data["EMA350"].iloc[-1]
        ema25_last = data["EMA25"].iloc[-1]
        aph = data['Close'].max()
        dt_aph = data.loc[data['Close'].idxmax(), 'Date']
        lastPx = data["Close"].iloc[-1]
        last_idx = len(data) - 1
        m1 = data.iloc[last_idx - 20]['Close']  # ~1 month (20 trading days)
        m3 = data.iloc[last_idx - 60]['Close']  # ~3 months
        distance = (aph - lastPx) / lastPx
        dataWatch.append({"TICKER": ticker,
                          "CONID": tickerDict[ticker],
                          "5Y_HIGH": aph,
                          "DT_HIGH": dt_aph,
                          "CLOSE": lastPx,
                          "DISTANCE": distance,
                          "M1_CLOSE":m1,
                          "M3_CLOSE":m3,
                          "XMA_ST":ema25_last,
                          "XMA_LT":ema350_last})

    df_watch = pd.DataFrame(dataWatch)
    return df_watch

######## may be put it in SQL handler as you manipulate sql object rather than yahoo




# # MSCI_EAFE = '^990300-USD-STRD'
# ticker = yf.Ticker(T_bill)
# df_Price = pd.DataFrame(ticker.history(period="5y"))
# # print(df_Price)
# print(ticker.get_actions())
# get_actions: returns the dates of dividend payouts and stock splits
# get_analysis: returns projections and forecasts of relevant financial variables, such as growth, earnings estimates, revenue estimates, EPS trends, etc. It returns the data available on yahoo finance under the “Analysis” tab (example)
# get_balance_sheet: returns historical yearly balance sheet data of a given company. It is analogous to the information available on Yahoo Finance under the “Balance Sheet” section of the “Financials” tab (example)
# get_calendar: returns important future dates and the datatype that is going to be disclosed on said date, in addition to their current estimates (low, average, and high).
# get_cashflow: returns the historical yearly cash flows of a given company. Analogous to the “Cash Flow” section under the “Financials” tab in Yahoo Finance (example)
# get_info: returns a dictionary with relevant information regarding a given company’s profile. Zip Code, industry sector, employee count, summary, city, phone, state, country, website, address, etc.
#get_earnings_dates return past and forward quarter earnings date with estimate reported and suprises









