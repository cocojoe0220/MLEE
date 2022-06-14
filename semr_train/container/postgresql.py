from .properties import postgre_dic
from psycopg2 import pool, InterfaceError, OperationalError
from log import logger

class XBSPostgre:
    def __init__(self):

        self.connectPool = pool.SimpleConnectionPool(1, 5, host=postgre_dic['host'], port=postgre_dic['port'],
                                                     user=postgre_dic['user'], password=postgre_dic['password'],
                                                     database=postgre_dic['database'])
        self.logger = logger

    def init_connect(self):
        self.connectPool = pool.SimpleConnectionPool(1, 5, host=postgre_dic['host'], port=postgre_dic['port'],
                                                     user=postgre_dic['user'], password=postgre_dic['password'],
                                                     database=postgre_dic['database'])

    def get_connect(self):
        conn = self.connectPool.getconn()
        cursor = conn.cursor()
        return conn, cursor

    def close_connect(self, conn, cursor):
        cursor.close()
        self.connectPool.putconn(conn)

    # 标点符号任务 #
    def search_punctuation_corpus(self, task_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute("SELECT content from paragraph_info where enable=0 and task_id=%s and status=1 and is_manual=0" % task_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # 分类任务 #
    def search_classify_corpus(self, task_id):

        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT label_id, paragraph_id, sen_context, sentence_id from sentence_info where enable=0 and task_id=%s  and status=1 and is_manual=0" % task_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_label_type(self, label_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute("SELECT label_type from label_info where id=%s and enable=0" % label_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # ===== #
    # 句子标签 #
    def search_data_from_sentence_info_two(self, label_id):

        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, sen_context, paragraph_id, sentence_id from sentence_info WHERE label_id=%s and enable=0 and sub_status=1 and is_sub_manual=0" % label_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_data_from_entity_info_two(self, sen_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT label_id, entity_context, entity_offset from entity_info WHERE sen_id=%s and enable=0" % sen_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # ===== #
    # 实体标签 #
    def search_entity_from_entity_info_three(self, label_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, entity_context, sen_id, entity_offset from entity_info WHERE label_id=%s and enable=0 and sub_status=1 and is_sub_manual=0" % label_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_attribute_from_entity_info_three(self, entity_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT label_id, entity_context, entity_offset from entity_info WHERE entity_id=%s and enable=0" % entity_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # =====================================================
    # 句子全局特征
    def search_sentence_from_sen_ids(self, sen_ids):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        sql = ', '.join(list(map(lambda x: "'%s'" % x, sen_ids)))
        cur.execute(
            "SELECT id, label_id, sen_context from sentence_info WHERE id in (%s) and enable=0" % sql) # and sub_status=1
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_entity_from_sen_id(self, sen_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, entity_context, entity_offset, label_id, is_sub_manual from entity_info WHERE sen_id=%s and enable=0" % sen_id)#  and sub_status=1
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_entity_from_entity_id(self, entity_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, entity_context, entity_offset, label_id, is_sub_manual from entity_info WHERE entity_id=%s and enable=0" % entity_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_sen_global_corpus_from_label_id(self, label_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT label_id, label_content, input_offset, sen_id from global_corpus WHERE label_id=%s" % label_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_sentence_feature_from_sen_id(self, sen_ids):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        sql = ', '.join(list(map(lambda x: "'%s'" % x, sen_ids)))
        cur.execute(
            "SELECT sen_id, sen_content, sen_feature from sentence_feature WHERE sen_id in (%s)" % sql)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # =====================================================
    # 段落全局特征
    def search_paragraph_from_paragraph_ids(self, paragraph_ids):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        sql = ', '.join(list(map(lambda x: "'%s'" % x, paragraph_ids)))
        cur.execute(
            "SELECT id from paragraph_info WHERE id in (%s) and enable=0 and status=1" % sql) #  and is_manual=0
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def search_sen_from_paragraph_id(self, paragraph_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, label_id, sen_context, sub_status, is_manual, is_sub_manual from sentence_info WHERE paragraph_id=%s and enable=0 and status=1 order by sentence_id" % paragraph_id)  # and sub_status=1
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # def search_sen_global_corpus_from_label_id(self, label_id):
    #     conn, cursor = self.get_connect()
    #     cur = conn.cursor()
    #     cur.execute(
    #         "SELECT label_id, label_content, input_offset, sen_id from global_corpus WHERE label_id=%s" % label_id)
    #     rows = cur.fetchall()
    #     self.close_connect(conn, cursor)
    #     return rows

    def search_entity_global_corpus_from_label_id(self, label_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT label_id, label_content, input_offset, paragraph_id from global_corpus WHERE label_id=%s" % label_id)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    def  search_paragraph_feature_from_paragraph_id(self, paragraph_ids):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        sql = ', '.join(list(map(lambda x: "'%s'" % x, paragraph_ids)))
        cur.execute(
            "SELECT paragraph_id, paragraph_content, paragraph_feature from paragraph_feature WHERE paragraph_id in (%s)" % sql)
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # def search_labels(self):
    #     conn, cursor = self.get_connect()
    #     cur = conn.cursor()
    #     cur.execute(
    #         "SELECT * from label_info WHERE task_id=251 and enable=0")
    #     rows = cur.fetchall()
    #     self.close_connect(conn, cursor)
    #     return rows

    def search_sen_by_paragraph_id(self, paragraph_id):
        conn, cursor = self.get_connect()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, label_id, sen_context, sentence_id from sentence_info WHERE paragraph_id=%s and enable=0 order by sentence_id" % paragraph_id)  # and sub_status=1
        rows = cur.fetchall()
        self.close_connect(conn, cursor)
        return rows

    # 断线重连
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

postgre = XBSPostgre()
