import cx_Oracle as oracle
from util import utility
import os
from datetime import timedelta
import time

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'


class Oracle(utility.Database):

    def __init__(self):
        utility.Database.__init__(self, oracle)

    @utility.connect
    def selectTopicClassificationFromRoot(self, brandPK):
        dic = {}
        for pk, name in self.select("select pk, name from comment_category WHERE deepth = 1 and is_delete = 1 and brand_PK = '%s'" % brandPK):
            print("name = " + name)
            print("pk = " + pk)
            dic[name] = pk
        return dic

    @utility.connect
    def selectBrand(self):
        pkMap = {}
        for pk, brand_name in self.select("select pk, brand_name from Brand WHERE is_delete = 1"):
            print("pk = " + pk)
            print("brand_name = " + brand_name)
#             pkMap[pk] = brand_name
            assert brand_name not in pkMap

            pkMap[brand_name] = pk
        return pkMap

    def selectParentUpward(self, pk):
        pks = []
        names = []
        parentpk = None
        while pk != None:
            for name, _parent_pk in self.select("select name, parent_pk from comment_category WHERE pk = '%s'" % pk):
                names.append(name)
                parentpk = _parent_pk

            if parentpk == None:
                break

            pks.append(pk)
            pk = parentpk
            parentpk = None

        pks = pks[::-1]
        names = names[::-1]
        return pks, names

    @utility.connect
    def selectGoodsComment(self, table, end, brand_pk=None, comment_level=None, platform_pk=None, maximum=10000):
        print('def selectGoodsComment(self, table, end, brand_pk=None, comment_level=None, platform_pk=None, maximum=10000):')
        arr = []
        start = end - timedelta(days=1)

        start = start.strftime("%Y-%m-%d %H:%M:%S")
        end = end.strftime("%Y-%m-%d %H:%M:%S")

#         condition = "from " + table + " WHERE reply_time < to_date('%s', 'yyyy-mm-dd hh24:mi:ss') and reply_time >= to_date('%s', 'yyyy-mm-dd hh24:mi:ss') and platform_pk = '%s'" % (end, start, platform_pk)
        clause = "from %s, GOODS_COMMENT_CATEGORY where " % table
        condition = []
        condition.append('%s.PK = GOODS_COMMENT_CATEGORY.GOODS_COMMENT_PK' % table)
        if platform_pk is not None:
            condition.append("%s.platform_pk = '%s'" % (table, platform_pk))
#         condition.append("rownum<=100")
        if brand_pk is not None:
            condition.append("%s.brand_pk = '%s'" % (table, brand_pk))

        if comment_level is not None:
            condition.append("%s.comment_level = '%s'" % (table, comment_level))

        clause += ' and '.join(condition)
        sql = "select %s.pk, %s.COMMENT_CONTENT, %s.brand_pk, %s.comment_level, GOODS_COMMENT_CATEGORY.COMMENT_CATEGORY_PK " % (table, table, table, table) + clause
        print("sql: " + sql)
        sqlCount = "select count(*) " + clause
        print("sql: " + sqlCount)

        for cnt, *_ in self.select(sqlCount):
            print("cnt = %d" % cnt)

        class Instance:

            def __init__(self, content, brand_pk, sentiment):
                self.content = content
                self.brand_pk = brand_pk
                self.sentiment = sentiment
                self.label = []

        dic = {}

        cnt = 0
        for pk, content, brand_pk, sentiment, COMMENT_CATEGORY_PK in self.select(sql):
            _, topicArr = self.selectParentUpward(COMMENT_CATEGORY_PK)

            if not topicArr:
                continue

            if pk not in dic:
                dic[pk] = Instance(content, brand_pk, sentiment)

            dic[pk].label.append(topicArr)
            cnt += 1
            if cnt >= maximum:
                break

        for pk in dic:
            inst = dic[pk]
            print("content =", inst.content)

            print("brand_pk =", inst.brand_pk)

            print("sentiment =", inst.sentiment)

            from classification.hierarchical.paragraph import Hierarchy, ParagraphInstance
            paragraphInstance = ParagraphInstance(None, inst.content, Hierarchy().convert(inst.label))

            arr.append((inst.brand_pk, inst.sentiment, paragraphInstance))
        return arr

    @utility.connect
    def selectTopicClassificationFromParent(self, classPK):
        dic = {}
        for pk, name in self.select("select pk, name from comment_category WHERE parent_pk = '%s'" % classPK):
            dic[name] = pk
        return dic


instance = Oracle()

if __name__ == '__main__':

    with instance:
        start = time.time()
        for args in instance.select("""
                select MXJGQM__GOODS_COMMENT.PK   
                from GOODS_COMMENT_CATEGORY, MXJGQM__GOODS_COMMENT 
                where MXJGQM__GOODS_COMMENT.PK = GOODS_COMMENT_CATEGORY.GOODS_COMMENT_PK"""):
            print(args)
        print('cost =', time.time() - start)

#     with instance:
#         start = time.time()
#         for args in instance.select("""
#                 select count(*)
#                 from MXJGQM__GOODS_COMMENT
#                 where (select count(*) from GOODS_COMMENT_CATEGORY
#                 where MXJGQM__GOODS_COMMENT.PK = GOODS_COMMENT_CATEGORY.GOODS_COMMENT_PK
#                 ) > 0 """):
#             print(args)
#         print('cost =', time.time() - start)

