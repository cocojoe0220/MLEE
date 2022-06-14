from .properties import postgre_dic
from psycopg2 import pool, InterfaceError, OperationalError
from log import logger

class XBSPostgre:
    def __init__(self):
        self.connectPool = pool.SimpleConnectionPool(5, 200, host=postgre_dic['host'], port=postgre_dic['port'],
                                                     user=postgre_dic['user'], password=postgre_dic['password'],
                                                     database=postgre_dic['database'])
        self.logger = logger

    def init_connect(self):
        self.connectPool = pool.SimpleConnectionPool(5, 200, host=postgre_dic['host'], port=postgre_dic['port'],
                                                     user=postgre_dic['user'], password=postgre_dic['password'],
                                                     database=postgre_dic['database'])

    def get_connect(self):
        conn = self.connectPool.getconn()
        cursor = conn.cursor()
        return conn, cursor

    def close_connect(self, conn, cursor):
        cursor.close()
        self.connectPool.putconn(conn)

    def close_all(self):
        self.connectPool.closeall()

    def search_algo_type_info(self, algo_type, task_id, global_version):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute("SELECT id, label_id from train_task_info where algo_type=%s and task_id=%s and train_type=1 and enable=0 and status=2 and version='%s'" % (algo_type, task_id, global_version)) # algo_type=%s and   % algo_type
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_label_id_info(self, label_id, task_id, global_version):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute("SELECT id, label_id, algo_type from train_task_info where label_id=%s and task_id=%s and train_type=1 and enable=0 and status=2 and version='%s'" % (label_id, task_id, global_version))  # algo_type=%s and   % algo_type
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def test_connect(self):
        try:
            conn, cursor = self.get_connect()
            cur = conn.cursor()
            cur.execute("SELECT * from label_info WHERE enable=0")
            self.close_connect(conn, cursor)
        except InterfaceError:
            self.logger.info("reconncect")
            self.init_connect()
        except OperationalError:
            self.logger.info("reconncect")
            self.init_connect()
        except pool.PoolError:
            self.logger.info("reconncect")
            self.init_connect()

    def search_global_version_model(self, task_id, global_version):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, label_id, algo_type from train_task_info where task_id=%s and train_type=1 and enable=0 and status=2 and version='%s'" % (task_id, global_version))  # algo_type=%s and   % algo_type
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows


postgre = XBSPostgre()