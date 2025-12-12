from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import numpy as np
import datetime
import os
import math
from functools import reduce
from typing import Dict, List, Tuple, Union
from dateutil.relativedelta import relativedelta

### _connListSQL should be a list = [login,password,url,dbName]


class SqlManip(object):
    def __init__(self,_connListSQL):
        self._connListSQL = _connListSQL

    def set_connListSQL(self, connList):
        self._connListSQL = connList

    def get_connListSQL(self):
        return self._connListSQL

    def server_connection(self):
        '''Create the connection for SQL dataBase'''
        cnx = None
        try:
            cnx = create_engine('mssql+pyodbc://' + self._connListSQL[0] + ':'+ self._connListSQL[1] + '@' + self._connListSQL[2] + '/' + self._connListSQL[3] + 
                                '?driver=xxxxxxxxxxxx',use_setinputsizes=False)
            cnx.connect()
        except SQLAlchemyError as e:
            print(e)
            os.system('pause')
            # logs(error)
        return cnx

    def conn(self):
        '''Create connection object to manipulate SQL table'''
        conn = None
        try:
            conn = self.server_connection().connect()
        except SQLAlchemyError as e:
            print(e)
            # logs(error)
            os.system('pause')
        return conn

    def cursor(self, _statement):
        '''Create a cursor to execute SQL procedure/query'''
        cursor = self.server_connection().raw_connection().cursor()
        try:
            cursor.execute(_statement)
        except SQLAlchemyError as e:
            print(e)
            # logs(error)
            os.system('pause')
        cursor.commit()
        cursor.close()

    def deleteSimple(self, _table, _schema=''):
        '''delete an entire table from SQL'''
        if _schema == '':
            deleteStatement = 'DELETE FROM '+_table
        else:
            deleteStatement = 'DELETE FROM '+_schema+'.'+_table
        self.cursor(deleteStatement)

    def importData(self,_df,_table, _schema='', _exists='append'):
      '''Import dataframe as sqltable'''
        try:
            if _schema == '':
                _df.to_sql(_table, self.conn(), if_exists=_exists, index=False)

            else:
                _df.to_sql(_table, self.conn(), schema=_schema, if_exists=_exists, index=False)

        except SQLAlchemyError as e:
            print(e)
            # logs(error)
            os.system('pause')

    def uploadTable(self, _path, _table, _exists='append'):
        '''Upload to sql already formated excel tables/ excel file must have the same name as the table'''
        df = pd.read_excel(_path + _table + '.xlsx')
        # check special date format:
        cols_list,dateStr = df.columns.values,'DATE'
        col_date = [col for col in cols_list if dateStr.lower() in col]
        if not col_date:
            pass
        else:
            myLib.apply_dates(col_date, df, '%Y-%m-%d', yearfirst_=True)
        print(df)
        cond = input('Upload?: ( ENTER | N )  ')
        if cond.lower() != 'n':
            self.importData(df, _table, _exists=_exists)
            print('Table ' + _table + ' has been uploaded')
        else:
            print('Upload aborted, please review the excel table you are trying to upload')

    def fast_engine(self):
        '''Create the engine for SQL dataBase'''
        engine = None
        try:
            engine = create_engine(
                'mssql+pyodbc://' + self._connListSQL[0] + ':' + self._connListSQL[1] + '@' + self._connListSQL[2] + '/' + self._connListSQL[3] + '?driver=ODBC+Driver+17+for+SQL+Server',
                fast_executemany=True)
        except SQLAlchemyError as e:
            print('Error encountered:\n')
            error = str(e.__dict__['orig'])
            # logs(error)
            print(error)
            os.system('pause')
        return engine

    def cnx_fast(self):
        '''Create connection object to manipulate SQL DB'''
        conn = self.fast_engine().connect()
        return conn

    def fast_insert_database(self, _df, tableName, _schema='', _exists='append'):
        '''INSERT A whole df inside database using the fast method'''
        try:
            if _schema == '':
                _df.to_sql(tableName, self.cnx_fast(), if_exists=_exists, index=False)
            else:
                _df.to_sql(tableName, self.cnx_fast(), schema=_schema, if_exists=_exists, index=False)
        except SQLAlchemyError as e:
            print('\nError encountered:\n')
            error = str(e.__dict__['orig'])
            # logs(error)
            print(error)
            os.system('pause')

    def fast_upload_df(self,  _df, tableName, _schema='',chunkSize=500, _exists='append'):
        '''
        UPLOAD DF  TO SQL DATABASE TO POPULATE TABLES in a very fast way 
        ensure the date columns are formatted as date format before uploading
        the df if large will be divided by chunksize in order to speed up the process
        '''
        # list_df = np.array_split(_df, math.ceil(len(_df) / chunkSize))
        list_df = []
        for chunk_idx in np.array_split(_df.index, math.ceil(len(_df) / chunkSize)):
            df_chunk = _df.loc[chunk_idx]
            list_df.append(df_chunk)

        for i in range(len(list_df)):
            self.fast_insert_database(list_df[i], tableName, _schema, _exists)
