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


def get_validTickersYF(dictTick):
    '''Return a dict of Valid and Invalid tickers for Yahoo
    insert a dict {yTicker:InstrumentName}'''
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
    '''lstConst should be a lol with the ticker yahoo in each list being [0] position'''
    dictSelected = {}
    countSel, validImp = 0, len(lstConst)
    for lstValue in tqdm(lstConst, total=len(lstConst), desc="Processing selection by setting FCF parameter"):
        instr = Stocks(lstValue[0])
        if instr.validFCF(mode_):
            countSel += 1
            cond = 1
        else:
            cond=0
        dictSelected[lstValue[0]] = {"NAME": lstValue[6]}

    print(f"{countSel}/{validImp} passed the first selection filter")
    df = pd.DataFrame.from_dict(dictSelected, orient="index")

    return df

def getEarningsDtConst(lstConst_):
    '''lstConst should be a lol with the ticker yahoo in each list being [0] position'''
    dictE = {}
    for const in tqdm(lstConst_, total = len(lstConst_), desc="Processing Earnings Date for Constituents"):
        instr = Stocks(const[0])
        lastE = instr.get_dtInfo('mostRecentQuarter')
        nxtFiscE = instr.get_dtInfo('nextFiscalYearEnd')
        nxtEannounce = instr.get_earnings_dt()
        dictE[const[0]] = {"LAST_REPORT_DATE": lastE,
                           "NXT_FISC_YE_DATE": nxtFiscE, 'NXT_ANN_DATE': nxtEannounce}

    df = pd.DataFrame.from_dict(dictE, orient="index")
    return df









