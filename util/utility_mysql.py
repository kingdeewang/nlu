from mysql.connector import errorcode

import mysql.connector
from util import utility
import re

from util.utility import connect
from assertpy.assertpy import assert_that


class MySQL(utility.Database):

    def __init__(self):
        utility.Database.__init__(self, mysql.connector)
        
#         with self:
#             cursor = self.cursor
#             cursor.execute('SET NAMES UTF8')

    def create_table_cn_qatype(self):
    #     https://dev.mysql.com/doc/refman/8.0/en/string-types.html
        table_description = (
            "CREATE TABLE `cn_qatype` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `sentence` varchar(128) NOT NULL unique,"
            "  `label` enum('QUERY','REPLY') NOT NULL,"
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB")

        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

    # use the following command line to view the created table;
    # show create table corpus.cn_qatype;

        insertor = ("INSERT INTO cn_qatype (sentence, label) VALUES (%s, %s)")

        dic = utility.readFolder(utility.corpusDirectory + 'cn/qatype', '.data')

        for label, sentences in dic.items():
            for sentence in sentences:
                try:
                    cursor.execute(insertor, (sentence, label))
                except mysql.connector.Error as err:
                    if err.errno == errorcode.ER_DATA_TOO_LONG:
                        print(label, 'sentence is too long, longer than 128 characters')
                        print(sentence)
                    elif err.errno == errorcode.ER_DUP_ENTRY:
                        ...
                    else:
                        print(err.msg)

        self.commit()

    def create_table_cn_phatics(self):
        table_description = (
            "CREATE TABLE `cn_phatics` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `sentence` varchar(128) NOT NULL unique,"
            "  `label` enum('NEUTRAL','PERTAIN') NOT NULL,"
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB")
        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

    # use the following command line to view the created table;
    # show create table corpus.cn_phatics;

        insertor = ("INSERT INTO cn_phatics "
               "(sentence, label) "
               "VALUES (%s, %s)")

        dic = utility.readFolder(utility.corpusDirectory + 'cn/phatics', '.data')

        for label, sentences in dic.items():
            with open(label + '.txt', 'w', encoding='utf8') as file:
                for sentence in sentences:
                    try:
                        cursor.execute(insertor, (sentence, label))
                    except mysql.connector.Error as err:
                        if err.errno == errorcode.ER_DATA_TOO_LONG:
                            print(label, 'sentence is too long, longer than 128 characters')
                            print(sentence)
                            print(sentence, file=file)
                        elif err.errno == errorcode.ER_DUP_ENTRY:
                            ...
                        else:
                            print(err.msg)

        self.commit()

    def create_table_cn_paraphrase(self):
        table_description = (
            "CREATE TABLE `cn_paraphrase` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `x` varchar(64) NOT NULL,"
            "  `y` varchar(64) NOT NULL,"
            "  `score` int(3) NOT NULL,"
            "  `training` bool default(1) NOT NULL,"
            "  unique index(`x` ,`y`),"
            "  CONSTRAINT `cn_paraphrase_check` CHECK(x < y), "
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin")
        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")
        self.commit()

    @connect
    def create_table_en_nli(self):
        table_description = (
            "CREATE TABLE `en_nli` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `x` varchar(64) NOT NULL,"
            "  `y` varchar(64) NOT NULL,"
            "  `score` int(3) NOT NULL,"
            "  `training` bool default(1) NOT NULL,"
            "  unique index(`x` ,`y`),"
            "  CONSTRAINT `en_nli_check` CHECK(x < y), "
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin")
        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")
        self.commit()

    @connect
    def create_table_cn_ner(self):
        table_description = ("""
            CREATE TABLE `cn_ner` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `sentence` varchar(64) NOT NULL,
  `ner` varchar(128) NOT NULL,
  `category` varchar(32) NOT NULL,
  `training` tinyint(1) NOT NULL DEFAULT '1',
  PRIMARY KEY (`id`),
  UNIQUE KEY `sentence` (`sentence`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci            
            """)
        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")
        self.commit()

    def insert_into_paraphrase(self, x, y, score, training=True):
        insertor = "INSERT INTO cn_paraphrase (x, y, score, training) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE x=VALUES(x), y=VALUES(y), score=VALUES(score), training=VALUES(training)"
        cursor = self.cursor
        try:
            cursor.execute(insertor, (x, y, score, training))
        except mysql.connector.Error as err:
            print(err.msg)
        self.commit()

    def insert_into_paraphrase_training_instances(self):
        insertor = "INSERT INTO cn_paraphrase (x, y, score) VALUES (%s, %s, %s)"

        dic = utility.readFolder(utility.corpusDirectory + 'cn/paraphrase', '.data')
        cursor = self.cursor
        for label, sentences in dic.items():
            with open(label + '.txt', 'w', encoding='utf8') as file:
                for sentence in sentences:
                    try:
                        print(sentence, '=', label)
                        x, y = sentence.split('/')
                        x = x.strip()
                        y = y.strip()

                        socre = int(label[2:])
                        cursor.execute(insertor, (x, y, socre))
                    except mysql.connector.Error as err:
                        if err.errno == errorcode.ER_DATA_TOO_LONG:
                            print(label, 'sentence is too long, longer than 128 characters')
                            print(sentence)
                            print(sentence, file=file)
                        elif err.errno == errorcode.ER_DUP_ENTRY:
                            ...
                        else:
                            print(x, y)
                            print(err.msg)
            self.commit()

    def insert_paraphrase_evaluate_instances(self):
        insertor = "INSERT INTO cn_paraphrase (x, y, score, training) VALUES (%s, %s, %s, False)"
        cursor = self.cursor
        for line in utility.Text(utility.corpusDirectory + 'cn/paraphrase/evaluate.txt'):
            print(line)
            m = re.compile('(.+) / (.+) = 0\.(\d\d)').fullmatch(line)
            assert m
            x, y, socre = m.groups()
            if socre.startswith('0'):
                print("socre.startswith('0'): ", socre)

            socre = int(socre)
            try:

                cursor.execute(insertor, (x, y, socre))
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DATA_TOO_LONG:
                    print(line, 'sentence is too long, longer than 128 characters')
#                     print(sentence, file=file)
                elif err.errno == errorcode.ER_DUP_ENTRY:
                    ...
                else:
                    print("('%s', '%s')" % (x, y))
                    print(err.msg)

        self.commit()

    def create_table_cn_structure(self):
        table_description = (
            "CREATE TABLE `cn_structure` ("
            "  `id` int(11) NOT NULL,"
            "  `infix` varchar(2048) NOT NULL,"
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB")
        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")
        self.commit()

    def insert_into_cn_structure(self):
        insertor = "INSERT INTO cn_structure (id, infix) VALUES (%s, %s)"
        cursor = self.cursor
        cursor.execute('delete from cn_structure')
        for count, infix in enumerate(utility.Text(utility.corpusDirectory + 'cn/dep.txt')):
            print(count)

            try:
                cursor.execute(insertor, (count, infix))
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DATA_TOO_LONG:
                    print(infix, 'sentence is too long, longer than 128 characters')
#                     print(sentence, file=file)
                elif err.errno == errorcode.ER_DUP_ENTRY:
                    ...
                else:
                    print(err.msg)

            if count % 10000 == 0:
                self.commit()

        self.commit()

    @connect
    def select_cn_structure(self, limit=None, shuffle=False):
        sql = "select infix from cn_structure order by "
        if shuffle:
            sql += 'rand() '
        else:
            sql += 'id '

        if limit:
            sql += 'limit %d' % limit

        return [infix for infix, *_ in self.select(sql)]

    @connect
    def select_cn_paraphrase(self):
        dic = []
        for _ in range(101):
            dic.append(set())

        mapTest = []
        for _ in range(101):
            mapTest.append(set())

        mapLookup = {}
        mapLookupTest = {}

        from util.corpus import ParaphraseInstance
        for x, y, score, training in self.select('select x, y, score, training from cn_paraphrase'):
            paraphraseInstance = ParaphraseInstance(x, y)
            paraphraseInstance.score = score

            if training:
                dic[score].add(paraphraseInstance)
                mapLookup[paraphraseInstance] = score
            else:
                mapTest[score].add(paraphraseInstance)
                mapLookupTest[paraphraseInstance] = score
        return dic, mapLookup, mapTest, mapLookupTest

    @connect
    def update_cn_structure(self, dic):
        self.execute("insert into cn_structure (id, infix) values " + ','.join("(%d,'%s')" % t  for t in dic.items()) + "ON DUPLICATE KEY UPDATE id=VALUES(id), infix=VALUES(infix)")

    def delete_last_cn_structure(self):
        cursor = self.cursor
        cursor.execute("delete from cn_structure where id in (select * from (select max(id) from cn_structure) as t)")
        self.commit()

    def create_table_cn_regulation(self):
        cursor = self.cursor
        table_description = (
            "CREATE TABLE `cn_regulation` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `infix` varchar(128) NOT NULL,"
            "  unique index(`infix`),"
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB")

        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

        self.commit()

    def update_cn_regulation(self, infix):
        insertor = "INSERT INTO cn_regulation (infix) VALUES (%s)"
        cursor = self.cursor
        try:
            cursor.execute(insertor, (infix,))
            self.commit()
        except mysql.connector.Error as err:
            print(err.msg)
            if err.errno == errorcode.ER_DATA_TOO_LONG:
                print(infix, 'sentence is too long, longer than 128 characters')
#                     print(sentence, file=file)
            elif err.errno == errorcode.ER_DUP_ENTRY:
                return False
            else:
                print(err.msg)
            return False
        return True

    def insert_into_cn_regulation(self):
        insertor = "INSERT INTO cn_regulation (infix) VALUES (%s)"
        cursor = self.cursor
        cursor.execute('delete from cn_regulation')
        for infix in utility.Text(utility.corpusDirectory + 'cn/err.txt'):

            try:
                cursor.execute(insertor, (infix,))
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_DATA_TOO_LONG:
                    print(infix, 'sentence is too long, longer than 128 characters')
#                     print(sentence, file=file)
                elif err.errno == errorcode.ER_DUP_ENTRY:
                    ...
                else:
                    print(err.msg)

        self.commit()

    def delete_from_cn_regulation(self, infix):
        cursor = self.cursor
        try:
            cursor.execute("delete from cn_regulation where infix = '%s'" % infix)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_DATA_TOO_LONG:
                print(infix, 'sentence is too long, longer than 128 characters')
#                     print(sentence, file=file)
            elif err.errno == errorcode.ER_DUP_ENTRY:
                ...
            else:
                print(err.msg)

        self.commit()

    @connect
    def select_cn_regulation(self):
        return [infix for infix, *_ in self.select("select infix from cn_regulation")]

    @connect
    def select_simplify(self):
        return {capitalized: substituent for capitalized, substituent in self.select("select capitalized, substituent from simplify")}

    @connect
    def create_table_semantic(self):
        sql = \
        """
        CREATE TABLE `tbl_semantic_log` (
          `keywords` varchar(200) NOT NULL,
          `category` varchar(40) NOT NULL,
          `semantic` varchar(500) DEFAULT NULL,
          `updatetime` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          `operator` varchar(40) DEFAULT NULL,
          PRIMARY KEY (`keywords`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='corpus for semantic classification'
        """
        self.execute(sql)

    @connect
    def execute_semantic(self):
        cursor = self.cursor

        cursor.execute("select count(DISTINCT  category) from tbl_semantic_log")
        yield from cursor

    @connect
    def select_semantic_save(self):
        dic = {}
        for keywords, category in self.select("select keywords, category from tbl_semantic_log"):
            if category not in dic:
                dic[category] = []
            dic[category].append(keywords)

        for category in dic:
            file = utility.corpusDirectory + 'cn/semantic/' + category + '.txt'
            print("writing ", file)
            utility.Text(file).write(dic[category])

    @connect
    def select_semantic(self, category=None, training=True):
        if category:
            sql = "select keywords, context, class from tbl_semantic_log where category = '%s' and training = %s" % (category, training)
        else:
            sql = "select keywords, context, category from tbl_semantic_log where training = %s" % training
        dic = {}
        for keywords, context, category in self.select(sql):
            if category not in dic:
                dic[category] = []

            dic[category].append((keywords, context))

        return dic

    @connect
    def select_semantic_flat(self, training=True):
        sql = "select sentence, context, service from cn_semantic where training = %s" % training
        dic = {}
        for sentence, context, service in self.select(sql):
            if service not in dic:
                dic[service] = []

            dic[service].append((sentence, context))

        return dic
    
    @connect
    def select_ner(self, category, training):
        if category:
            sql = "select sentence, ner from cn_ner where category = '%s' and training = %s" % (category, training)
        else:
            sql = "select sentence, ner from cn_ner where training = %s" % training
        array = []
        for sent, ner in self.select(sql):
            array.append((sent, eval(ner)))
#             array.append((sent, json.loads(ner)))

        return array

    @connect
    def select_semantic_search(self, keywords, category=None):
        from classification import semanticClassifier
        cnt = 0
        sql = "select keywords, category from tbl_semantic_log where keywords like '%%%s%%'" % keywords
        if category is not None:
            sql += " and category = '%s'" % category
            assert category in semanticClassifier.instance.label
        print(sql)
        res = []
        for keywords, category in self.select(sql):
            print(keywords, '=', category)
            res.append(keywords)
            cnt += 1

        print(',\n'.join("'%s'" % keyword  for keyword in res))
        print('cnt =', cnt)

    @connect
    def insert_into_semantic(self, array=None):

        if array is None:
            sql = "select keywords, context, category, code from tbl_semantic_log"
            import requests
            r = requests.post('http://111.206.59.9:8000/select/' + sql)

            array = eval(r.text)

#         dic = utility.readFolder('/home/deeplearning/text-classfication/data', '.txt')
#         dic = utility.readFolder(utility.corpusDirectory + 'cn/semantic', '.txt')
        insertor = "INSERT INTO tbl_semantic_log(keywords, context, category, code) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE keywords=VALUES(keywords), context=VALUES(context), category=VALUES(category), code=VALUES(code)"
        from classification import semanticClassifier_flat

        cursor = self.cursor

        for keywords, context, category, code in array:
#             if category not in semanticClassifier.instance.label:
#                 print(category, ' not in semanticClassifier.instance.label')
#                 continue

#             assert code in semanticClassifier_flat.instance.label, str(semanticClassifier_flat.instance.label) + " : " + code
            if code not in semanticClassifier_flat.instance.label:
                print(keywords, context, category, code)
            assert context in semanticClassifier_flat.instance.context, str(semanticClassifier_flat.instance.context) + " : " + context

            try:
                cursor.execute(insertor, (keywords, context, category, code))
            except mysql.connector.Error as err:
                print(err.msg)

        self.commit()
        print('successfully commited')

    @connect
    def insert_into_ner(self, array=None):

        insertor = "INSERT INTO cn_ner(sentence, ner) VALUES (%s, %s) ON DUPLICATE KEY UPDATE sentence=VALUES(sentence), ner=VALUES(ner)"

        cursor = self.cursor

        for sent, ner in array:

            try:
                cursor.execute(insertor, (sent, ner))
            except mysql.connector.Error as err:
                print(err.msg)

        self.commit()
        print('successfully commited')

    @connect
    def update_semantic_like(self, keywords, category, category_modified):
        sql = "update tbl_semantic_log set category='%s' WHERE keywords like '%%%s%%' and category = '%s'" % (category_modified, keywords, category)
        print(sql)

        from classification import semanticClassifier
        assert category_modified in semanticClassifier.instance.label

        self.execute(sql)

    @connect
    def update_semantic_single(self, keywords, category, training=True):
        sql = "INSERT INTO tbl_semantic_log(keywords, context, category, code, training) VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE keywords=VALUES(keywords), context=VALUES(context), category=VALUES(category), code=VALUES(code), training=VALUES(training)"
        print(sql)

        m = re.compile(r'(.+) *\| *([a-z]+)').fullmatch(keywords)
        if m:
            keywords, context = m.groups()
        else:
            context = ''

        print('keywords =', keywords)
        print('context  =', context)
        print('training  =', training)

        from classification import semanticClassifier_flat
        code = None
        for key, array in semanticClassifier_flat.instance.dic.items():
            if category in array:
                code = category
                category = key
                print('category =', category)
                print('code =', code)
                break

        assert_that(code in semanticClassifier_flat.instance.label).is_true()
        assert_that(context in semanticClassifier_flat.instance.context).is_true()
        assert code is not None

        self.execute(sql, (keywords, context, category, code, training))

    @connect
    def delete_semantic(self, keywords):
        sql = "delete from tbl_semantic_log where keywords = %s"
        print(sql)

        print('keywords =', keywords)

        self.execute(sql, (keywords,))

    @connect
    def update_semantic_array(self, keywords, category):
        sql = "update tbl_semantic_log set category='%s' WHERE keywords in (%s)" % (category, ', '.join("'%s'" % key for key in keywords))
        print(sql)

        from classification import semanticClassifier
        assert category in semanticClassifier.instance.label

        self.execute(sql)

    def create_table_simplify(self):
        table_description = (
            "CREATE TABLE `simplify` ("
            "  `id` int(11) NOT NULL AUTO_INCREMENT,"
            "  `capitalized` char(1) NOT NULL,"
            "  `substituent` char(1) NOT NULL,"
            "  `lang` char(2) NOT NULL,"
            "  unique index(`capitalized`),"
            "  PRIMARY KEY (`id`)"
            ") ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin")

# cn, en, fr, de, ru, ja, pu
        cursor = self.cursor
        try:
            print(table_description)
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

        self.commit()

    def insert_into_simplify(self):
        insertor = "INSERT INTO simplify (capitalized, substituent, lang) VALUES (%s, %s, %s)"
        cursor = self.cursor
        cursor.execute('delete from simplify')
        cursor.execute('alter table simplify AUTO_INCREMENT=0')

        for lang, lines in utility.readFolder(utility.modelsDirectory + 'simplify', '.txt').items():

            for line in lines:
                characters, substituent = re.compile("(\S+)\s*=>\s*(\S+)").fullmatch(line).groups()

                for capitalized in characters:
                    try:
                        cursor.execute(insertor, (capitalized, substituent, lang))
                    except mysql.connector.Error as err:
                        print(err.msg)

        self.commit()

    def update_cn_ner(self, sent, ner, category, training=1):
        sql = "INSERT INTO cn_ner(sentence, ner, category, training) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE sentence=VALUES(sentence), ner=VALUES(ner), category=VALUES(category), training=VALUES(training)"
        print(sql)

        print('sent =', sent)
        print('ner  =', ner)
        print('category =', category)
        print('training =', training)
#         print('code =', code)
        
        ner = str(ner)
        self.execute(sql, (sent, ner, category, training))

    @connect
    def select_cn_ner(self, sent):
        sql = "select sentence, ner, category from cn_ner where sentence = '%s'" % sent
#         print(sql)

        for args in self.select(sql):
            return args

    def update_cn_ner_code(self, sent, code):
        sql = "update cn_ner set code = %s where sentence = %s"
        print(sql)

        print('sent =', sent)
        
        print('code =', code)        

        self.execute(sql, (code, sent))

    @connect
    def update_cn_ner_change_label(self, old, new):
        sql = "select sentence, ner, category from cn_ner where ner like '%%''%s'':%%'" % old
        print(sql)

        array = []
        for sent, ner, category in self.select(sql):

            print('sent =', sent)
            print('ner  =', ner)
            print('category  =', category)
            ner = eval(ner)

            ner[new] = ner[old]
            del ner[old]

            array.append((sent, ner, category))

        for sent, ner, category in array:
            self.update_cn_ner(sent, ner, category)

    @connect
    def delete_cn_ner(self, sent):
        sql = "delete from cn_ner where sentence = %s"
        print(sql)

        print('sent =', sent)

        self.execute(sql, (sent,))

    @connect
    def write_log(self):
        with open(utility.corpusDirectory + 'cn/debug.ner.txt', 'w', encoding='utf8') as file:
            for sent, ner, category in instance.select("select sentence, ner, category from cn_ner"):
                ner = eval(ner)

                found = False

                _ner = not_contains(sent, ner, 'tag', '抖音')
                if _ner is not None:
                    ner = _ner
                    found = True
                
                _ner = not_contains(sent, ner, 'language', '印度')
                if _ner is not None:
                    ner = _ner
                    found = True
#                     
#                     if ner[]
                                    
#                                  if contains(ner, 'song', '一曲'):
#                 if contains(ner, 'artist', 'dj'):
#                     found = True
#                     
#                 if contains(ner, 'genre', '最近'):
#                     found = True
#                 if not_contains(sent, ner, 'genre', '流行'):
#                     found = True
                    
                if found :
                    if "'" in sent:
                        print("(\"%s\", %s, '%s')" % (sent, ner, category), file=file)
                    else:
                        print("('%s', %s, '%s')" % (sent, ner, category), file=file)
        
    @connect
    def ner_fragment(self, fragment, dic, category='music'):
        
        if isinstance(dic, str):
            dic = eval(dic)
            
        from sequence import nerecognizer
        fragment = fragment.lower()
        enum = eval('nerecognizer.Status%s' % category.upper())
        
        print(fragment)
        print(dic)
        
        fragment_inst = nerecognizer.Instance(enum, fragment, dic)
        
        arr = []
        for sentence, ner in instance.select("select sentence, ner from cn_ner where sentence like '%%%s%%' and category = '%s'" % (fragment, category)):
            sent = sentence.lower()
#                 if sent == '夜的钢琴曲好听吧':
#                     print(sent)
            found = False
            
            ner = eval(ner)
            for key, value in ner.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if v.startswith(' ') or v.endswith(' '):
                            value[i] = v.strip()
                            found = True
                else:
                    if value.startswith(' ') or value.endswith(' '):
                        ner[key] = value.strip() 
                        found = True
            
            incomplete = True
            
            inst = nerecognizer.Instance(enum, sent, ner)
            
            indices = []
            
            index = sent.index(fragment)
            
            while True:                    
                indices.append(index)
                try:
                    index = sent.index(fragment, index + len(fragment))
                except:
                    break
                
            for index in indices:
                end = index + len(fragment)
                if inst.tag[index: end] != fragment_inst.tag:
                    if (inst.tag[index].value & 1) == 0 and (end >= len(inst.tag) or (inst.tag[end].value & 1) == 0):      
                        for i in range(index, end):
                            inst.tag[i] = fragment_inst.tag[i - index]
                        found = True                    
                else:
                    incomplete = False
                
            if found :
                ner = enum.toDict(sentence, [tag.value for tag in inst.tag])                    
                
                arr.append({'sent' : sentence, 'ner' : ner, 'category' : category})
            elif incomplete:
                print(nerecognizer.string_tuple(sentence, ner, category))
                
        return arr
    
    @connect
    def ner_fragment_(self, fragment, dic, category='music'):
        log = utility.corpusDirectory + 'cn/debug.ner.txt'
        for line in utility.Text(log): 
            self.update_cn_ner(*eval(line.strip()))
        
        from sequence import nerecognizer
        fragment = fragment.lower()
        enum = eval('nerecognizer.Status%s' % category.upper())
        
        with open(log, 'w', encoding='utf8') as file:
            for sentence, ner in instance.select("select sentence, ner from cn_ner where sentence like '%%%s%%' and category = '%s' and training != -1" % (fragment, category)):
                sent = sentence.lower()
#                 if sent == '夜的钢琴曲好听吧':
#                     print(sent)
                found = False
                
                ner = eval(ner)
                for key, value in ner.items():
                    if isinstance(value, list):
                        for i, v in enumerate(value):
                            if v.startswith(' ') or v.endswith(' '):
                                value[i] = v.strip()
                                found = True
                    else:
                        if value.startswith(' ') or value.endswith(' '):
                            ner[key] = value.strip() 
                            found = True
                
                incomplete = True
                
                inst = nerecognizer.Instance(enum, sent, ner)
                fragment_inst = nerecognizer.Instance(enum, fragment, dic)
                
                indices = []
                
                index = sent.index(fragment)
                
                while True:                    
                    indices.append(index)
                    try:
                        index = sent.index(fragment, index + len(fragment))
                    except:
                        break
                    
                for index in indices:
                    end = index + len(fragment)
                    if inst.tag[index: end] != fragment_inst.tag:
                        if (inst.tag[index].value & 1) == 0 and (end >= len(inst.tag) or (inst.tag[end].value & 1) == 0):      
                            for i in range(index, end):
                                inst.tag[i] = fragment_inst.tag[i - index]
                            found = True                    
                    else:
                        incomplete = False
                    
                if found :                    
                    ner = enum.toDict(sentence, [tag.value for tag in inst.tag])
                    
                    print(nerecognizer.string_tuple(sentence, ner, category), file=file)
                elif incomplete:
                    print(nerecognizer.string_tuple(sentence, ner, category))
                    
            for fragment, ner in instance.select("select sentence, ner from cn_ner where sentence = '%s' and training = -1" % fragment):
                ner = eval(ner)
                if ner == dic:
                    return
            print(nerecognizer.string_tuple(fragment, dic, category, -1), file=file)

    @connect
    def ner_add_phrase(self, status, phrase):
        with open(utility.corpusDirectory + 'cn/debug.ner.txt', 'w', encoding='utf8') as file:
            for sent, ner, category in instance.select("select sentence, ner, category from cn_ner"):
                ner = eval(ner)

                found = False
                
                if isinstance(phrase, (set, list)):
                    for p in phrase:
                        _ner = not_contains(sent, ner, status, p)
                        if _ner is not None:
                            ner = _ner
                            found = True
                        
                else:
                    _ner = not_contains(sent, ner, status, phrase)
                    if _ner is not None:
                        ner = _ner
                        found = True
                
                if found :
                    if "'" in sent:
                        print("(\"%s\", %s, '%s')" % (sent, ner, category), file=file)
                    else:
                        print("('%s', %s, '%s')" % (sent, ner, category), file=file)

    @connect
    def ner_contains_phrase(self, status, phrase):
        with open(utility.corpusDirectory + 'cn/debug.ner.txt', 'w', encoding='utf8') as file:
            for sent, ner, category in instance.select("select sentence, ner, category from cn_ner"):
                ner = eval(ner)

                found = False
                if isinstance(phrase, (set, list)):
                    for p in phrase:
                        if contains(ner, status, p):
                            found = True
                else:
                    if contains(ner, status, phrase):
                        found = True
                
                if found :
                    if "'" in sent:
                        print("(\"%s\", %s, '%s')" % (sent, ner, category), file=file)
                    else:
                        print("('%s', %s, '%s')" % (sent, ner, category), file=file)

    @connect
    def write_log_sent(self):
        with open(utility.corpusDirectory + 'cn/debug.ner.txt', 'w', encoding='utf8') as file:
            for sent, ner, category in instance.select("select sentence, ner, category from cn_ner where sentence like '%继续下%'"):
                
                if "'" in sent:
                    print("(\"%s\", %s, '%s')" % (sent, ner, category), file=file)
                else:
                    print("('%s', %s, '%s')" % (sent, ner, category), file=file)

    @connect
    def ner_set(self, status):
        language_set = set()
        for sent, ner, category in instance.select("select sentence, ner, category from cn_ner where ner like '%%''%s'':%%'" % status):
            ner = eval(ner)
            language = ner[status]
            if isinstance(language, list):
                for l in language:
                    language_set.add(l)
            else:
                language_set.add(language)
        print(language_set)
                            
            
def value_contains(status, content):

    if isinstance(status, list):
        for a in status:
            if content in a.lower():
                return True
            
    else:                            
        if content in status.lower():
            return True
    return False


def value_full_match(value, content):

    if isinstance(value, list):
        for a in value:
            if content == a.lower():
                return True
            
    else:                            
        if content == value.lower():
            return True
    return False


def value_delete(value, content):

    if isinstance(value, list):
        for a in value:
            if content == a.lower():
                value.remove(content)
                if len(value) == 1:
                    return value[0]
                return value
            
    else:                            
        if content == value.lower():
            return None
    return value


def value_add(value, content):
    if isinstance(value, list):
        value.append(content)
        return value            
    elif value is None:                          
        return content
    else:
        return [value, content]


def contains(ner, status, content):
    if status not in ner:
        return False
    status = ner[status]
    if isinstance(status, list):
        for a in status:
            if content in a.lower():
                return True
            
    else:                            
        if content in status.lower():
            return True
    return False

    
def not_contains(sent, ner, status, c):     
    if c not in sent:
        return None
    
    for key, value in [*ner.items()]:
        if value_contains(value, c):
            if key == status:
                return None
            
            if not value_full_match(value, c):
                return None
            value = value_delete(value, c)
            if value is None:
                del ner[key]
            else:
                ner[key] = value
    
    if status not in ner:
        
        ner[status] = c        
        
        return ner
    
    value = ner[status]

    if isinstance(value, list):
        for a in value:
            if c in a.lower():
                return None
            
        value.append(c)
    else:                            
        if c in value.lower():
            return None
        
        ner[status] = [value, c] 
    return ner


instance = MySQL()

if __name__ == '__main__':
    
#     for line in utility.Text(utility.corpusDirectory + 'cn/semantic.txt'):
#         print(instance.select_cn_ner(line))

    with instance:  
        for line in utility.Text(utility.corpusDirectory + 'cn/semantic.txt'):
            instance.update_cn_ner(*eval(line.strip()))

#     arr = instance.ner_fragment('歌曲串烧', {"genre":"串烧"}, 'music')
#     for dic in arr:
#         print(dic)
