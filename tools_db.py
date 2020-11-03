
from dotmap import DotMap



def kw2sql__params(__joint=None, **kw):
    '''
    parse key_word to equations and params
    e.g,    INPUT:  x=x,y=y
            OUTPUT:['x=?','y=?'],[x,y]
    :param kw:
    :type kw:
    :return:
    :rtype:
    '''
    sqls = []
    params = []
    if len(kw) > 0:
        for k, v in kw.items():
            sqls.append(f'{k}=(?)')
            params.append(v)
    if __joint is not None:
        __joint = f' {__joint} '
        sqls = __joint.join(sqls)
    return sqls, params



def convert_sqlwhere__params(condition='', params=None, condition_kw_joint=None, **condition_kw):
    '''
    transform 'key2word' to 'where' string
    '<condition> <condition_kw_str>'
    :param condition:
    :type condition:
    :param params:
    :type params:
    :param condition_kw:
    :type condition_kw:
    :return: '<condition> <condition_kw>'
    :rtype:

    NOTE: I did't use the format string f'x={x}' directly, as it may be not safe for sql bug.
    '''

    # --- Preprocessing the condition
    if params is None:
        params = []
    if condition_kw_joint is None:
        condition_kw_joint = 'and'

    condition = condition.strip()
    condition_kw_str, params_kw = kw2sql__params(__joint=condition_kw_joint, **condition_kw)
    condition_kw_str = condition_kw_str.strip()

    # --- Combining condition and conidtion_kw
    if condition != '' and condition_kw_str != '':
        condition_tmp = condition.strip().lower()
        assert condition_tmp.endswith('and') or condition_tmp.endswith('or'), 'condition_partial should end with "and" or "or"'

    condition_final = ''
    if condition != '' or condition_kw_str != '':
        condition_final = f' where {condition} {condition_kw_str}'
    params = params + params_kw

    return condition_final, params

class DbHelper:
    def __init__(self, table, conn):
        self.table = table
        self.conn = conn

    def cnt(self, table='', condition='', params=None, condition_kw_joint='and', **condition_kw):
        if not table:
            table = self.table

        sqlwhere, params = convert_sqlwhere__params(condition, params, condition_kw_joint=condition_kw_joint, **condition_kw)
        cursor = self.conn.cursor()
        cursor.execute(f'select count(*) from {table} {sqlwhere}', params)
        return cursor.fetchone()[0]

    def exist(self, table='', condition='', params=[], **condition_kw):
        if not table:
            table = self.table
        return self.cnt(table, condition, params, **condition_kw) > 0

    def get(self, table='', return_type='default', return_kwargs={}, condition='', params=None, condition_kw_joint=None, **condition_kw):
        if not table:
            table = self.table
        sqlwhere, params = convert_sqlwhere__params(condition, params, condition_kw_joint=condition_kw_joint, **condition_kw)
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * from {table} {sqlwhere}", params)
        result_all = cursor.fetchall()
        if return_type == 'default':
            '''
            [
                item (list format),
                item (list format),
            ]
            '''
            return result_all
        elif return_type == 'dict':
            '''
            [
                item (dict format),
                item (dict format)
            ]
            '''
            for ind in range(len(result_all)):
                key2index = self.columns(table, 'key2index')
                result = result_all[ind]
                result_all[ind] = DotMap()
                for name,index in key2index.items():
                    result_all[ind][name] = result[index]
            return result_all
        elif return_type == 'key2dict':
            '''
            {
                <value of key of item>:<item (dict format)>,
                <value of key of item>:<item (dict format)>,
            }
            '''
            key2index = self.columns(table, 'key2index')
            ind_key_main = key2index[ return_kwargs['key'] ]
            results_dict = {}
            for result in result_all:
                results_dict[result[ind_key_main]] = DotMap()
                for name,index in key2index.items():
                    results_dict[result[ind_key_main]][name] = result[index]
            return results_dict
        else:
            raise NotImplementedError


    def get_dict(self, table='', condition_partial='', params=None, condition_kw_joint=None, **condition_kw):
        return self.get(table, return_type='dict', return_kwargs=dict(), condition=condition_partial, params=params, condition_kw_joint=condition_kw_joint, **condition_kw)

    def get_key2dict(self, key, table='', condition='', params=None, condition_kw_joint=None, **condition_kw):
        return self.get(table, return_type='key2dict', return_kwargs=dict(key=key), condition=condition, params=params, condition_kw_joint=condition_kw_joint, **condition_kw)

    # def show_all(self, **kwfilters):
    #     print('-------items------')
    #     for i in self.get(**kwfilters):
    #         print(i)
    #     print('-------items end------')


    def insert(self, table='', **kw):
        if not table:
            table = self.table
        cursor = self.conn.cursor()
        columns_sql = ",".join(kw.keys())
        values_placeholder = ",".join(["?"] * len(kw))
        cursor.execute(f'insert into {table} ({columns_sql}) values ({values_placeholder})', list(kw.values()))
        self.conn.commit()
        return self.conn.total_changes

    def update(self, update_kw, table='', condition='', params=None, condition_kw_joint=None, **condition_kw):
        if not table:
            table = self.table
        sql_set, params_set = kw2sql__params(__joint=',', **update_kw)
        sql_where, params_where = convert_sqlwhere__params(condition, params, condition_kw_joint=condition_kw_joint, **condition_kw)
        assert sql_where.strip() != '', 'Please make sure that you will update all the items, or please use 1=1'
        cursor = self.conn.cursor()
        cursor.execute(f'update {table} set {sql_set} {sql_where}', params_set+ params_where)
        self.conn.commit()
        return self.conn.total_changes


    def delete(self, table='', condition='', params=None, condition_kw_joint=None, **condition_kw):
        if not table:
            table = self.table
        sql_where, params_where = convert_sqlwhere__params(condition, params, condition_kw_joint=condition_kw_joint, **condition_kw)
        assert sql_where.strip() != '', 'Please make sure that you will delete all the items, or please use 1=1'
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE from {table} {sql_where}", params_where)
        self.conn.commit()
        return self.conn.total_changes

    def columns(self, table='', fmt='list'):
        if not table:
            table = self.table
        cursor = self.conn.cursor()
        cursor.execute(f'PRAGMA table_info({table})')
        columns = cursor.fetchall()
        if fmt == 'list':
            columns = [c[1] for ind, c in enumerate(columns)]
        elif fmt == 'key2index':
            columns = {c[1]:ind  for ind,c in enumerate(columns)}
        else:
            raise NotImplementedError
        return columns
