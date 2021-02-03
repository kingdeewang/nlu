from math import nan
import os
import re

from bert import tokenization, optimization
from util import utility, utility_mysql
import random
from assertpy.assertpy import assert_that
import numpy


class ParaphraseInstance:

    def __init__(self, left, right, score=nan, normalize=True):
        self.x = left
        self.y = right
        self.score = score
        if normalize :
            self.normalize()

    def copy(self):
        return ParaphraseInstance(self.x, self.y, self.score, normalize=False)

    def reverse(self):
        inst = ParaphraseInstance(self.y, self.x, self.score, normalize=False)
        if hasattr(self, 'source'):
            x, y = re.compile(' */ *').split(self.source)
            inst.source = y + ' / ' + x
        return inst

    def identity(self):
        inst_x = ParaphraseInstance(self.x, self.x, 1, normalize=False)
        inst_y = ParaphraseInstance(self.y, self.y, 1, normalize=False)
        if hasattr(self, 'source'):
            x, y = re.compile(' */ *').split(self.source)
            inst_x.source = x + ' / ' + x
            inst_y.source = y + ' / ' + y
        return inst_x, inst_y

    def normalize(self):
        if self.x > self.y:
            tmp = self.y
            self.y = self.x
            self.x = tmp

        return self

    def __hash__(self):
        return hash(self.x.lower()) + hash(self.y.lower()) * 31

    def __eq__(self, other):
        return self.x.lower() == other.x.lower() and self.y.lower() == other.y.lower()

    def __lt__(self, other):
        if self.x.lower() < other.x.lower():
            return True
        if self.x.lower() > other.x.lower():
            return False
        return self.y.lower() < other.y.lower()

    def __str__(self):
        if hasattr(self, 'source'):
            return self.source

        return self.x + " / " + self.y


def normalize(value):
    if isinstance(value, int):
        assert_that(value).is_less_than_or_equal_to(100)

        return value

    if isinstance(value, float):
        value = int(value * 100)
        assert_that(value).is_less_than_or_equal_to(100)
        return value

    if re.compile("[01](\.\d+)?").fullmatch(value):
        value = int(float(value) * 100)
        assert_that(value).is_less_than_or_equal_to(100)
        return value

    assert re.compile("\d+").fullmatch(value)
    value = int(value)
    assert value <= 100
    return value


def match(sent):
    # use greedy search instead
    m = re.compile("([^/]+)/([^=]+)(?:\s*=\s*(\S+))?").match(sent)
    assert m, sent
    x, y, value = m.groups()
    x = x.strip()
    y = y.strip()
    assert '�' not in x and '�' not in y and x.lower() != y.lower(), "%s / %s = %s, sent = %s" % (x, y, value, sent)

    return x, y, value


class Corpus:
    folder = utility.corpusDirectory + 'cn/paraphrase/'
    evaluate_file = utility.corpusDirectory + 'cn/paraphrase/evaluate.txt'

    def establishCorpusMySQL(self):
        self.map, self.mapLookup, self.mapTest, self.mapLookupTest = utility_mysql.instance.select_cn_paraphrase()

    def nerCorpus(self, category=None, training=1):
        return utility_mysql.instance.select_ner(category, training)

    def semanticCorpus(self, category=None, training=True):
        return utility_mysql.instance.select_semantic(category, training)

    def semanticCorpus_flat(self, training=True):
        return utility_mysql.instance.select_semantic_flat(training)

    def validateTrainingCorpus(self, validateSet):
        for score in validateSet:
            print('update file', Corpus.folder + score + '.data')
            if score in self.map:
                utility.Text(Corpus.folder + score + '.data').write(self.map[score])
            else:
                utility.Text(Corpus.folder + score + '.data').clear()

    def readFile(self, score, mp={}):
        bInitiallyEmpty = False if mp else True

        for sent in utility.Text(Corpus.folder + score + '.data'):
            x, y = re.compile(' */ *').split(sent)
#                 m = re.compile("(.+?)( *= *)(.+)").match(y)
# use greedy search instead
            m = re.compile("(.+?)( *= *)(\S+)").match(y)
            if m:
                y = m.group(1)
                value = m.group(3)
                try:
                    value = self.normalize(value)
                except:
                    print(sent, '\ninvalid score value =', value)
                    value = score
            else:
                value = score

            if value not in mp:
                mp[value] = set()

            paraphraseInstance = ParaphraseInstance(x, y)

            mp[value].add(paraphraseInstance)

        if score not in self.map:
            self.map[score] = set()

        if score not in mp:
            mp[score] = set()

        if self.map[score] != mp[score]:
            for e in self.map[score] - mp[score]:
                del self.mapLookup[e]
            for e in mp[score] - self.map[score]:
                self.mapLookup[e] = score

            self.map[score] = mp[score]

        del mp[score]
        if not bInitiallyEmpty or mp:
            print('update file', Corpus.folder + score + '.data')
            if score in self.map:
                utility.Text(Corpus.folder + score + '.data').write(self.map[score])
            else:
                utility.Text(Corpus.folder + score + '.data').clear()

        for score in list(mp.keys()):
            self.readFile(score, mp)

    def corpusSimilarity(self, x, y, score=None):

        if score is None:
            paraphraseInstance = ParaphraseInstance(x, y)

            if paraphraseInstance in self.mapLookup:
                return self.mapLookup[paraphraseInstance]

            if paraphraseInstance in self.mapLookupTest:
                print('in the test set:')
                return self.mapLookupTest[paraphraseInstance]

            return nan

        value = normalize(score)

        paraphraseInstance = ParaphraseInstance(x, y)

        oldSimilarity = nan

        if paraphraseInstance in self.mapLookupTest:
            oldSimilarity = self.mapLookupTest[paraphraseInstance]
            if oldSimilarity == value:
                return oldSimilarity

            self.mapLookupTest[paraphraseInstance] = value
            self.mapTest[oldSimilarity].remove(paraphraseInstance)
            self.mapTest[value].add(paraphraseInstance)

            utility_mysql.instance.insert_into_paraphrase(x, y, value, False)
            return oldSimilarity

        if paraphraseInstance in self.mapLookup:
            oldSimilarity = self.mapLookup[paraphraseInstance]
            if oldSimilarity == value:
                return oldSimilarity

            self.mapLookup[paraphraseInstance] = value
            self.map[value].add(paraphraseInstance)
            self.map[oldSimilarity].remove(paraphraseInstance)

            utility_mysql.instance.insert_into_paraphrase(x, y, value, True)
            return oldSimilarity

        if value not in self.map:
            self.map[value] = set()

        if isinstance(score, float):
            self.mapLookupTest[paraphraseInstance] = value
            self.mapTest[value].add(paraphraseInstance)
            utility_mysql.instance.insert_into_paraphrase(x, y, value, False)
        else:
            self.mapLookup[paraphraseInstance] = value
            self.map[value].add(paraphraseInstance)
            utility_mysql.instance.insert_into_paraphrase(x, y, value, True)

        return oldSimilarity

    def delete_from_training_corpus(self, paraphraseInstance, validateSet=None):
        oldSimilarity = self.mapLookup[paraphraseInstance]
        self.map[oldSimilarity].remove(paraphraseInstance)
        del self.mapLookup[paraphraseInstance]

        if validateSet is None:
            utility.Text(Corpus.folder + oldSimilarity + ".data").write(self.map[oldSimilarity])
        else:
            validateSet.add(oldSimilarity)
        return oldSimilarity

    def delete(self, x, y):
        paraphraseInstance = ParaphraseInstance(x, y)

        oldSimilarity = nan
        if paraphraseInstance in self.mapLookup:
            self.delete_from_training_corpus(paraphraseInstance)

        if paraphraseInstance in self.mapLookupTest:
            oldSimilarity = self.mapLookupTest[paraphraseInstance]
            del self.mapLookupTest[paraphraseInstance]

            self.saveTestSet()

        return oldSimilarity

    def saveTestSet(self):
        evaluateSample = list(self.mapLookupTest)
        evaluateSample.sort(key=lambda x : self.mapLookupTest[x], reverse=True)

        with open(self.evaluate_file, mode='w', encoding='utf8') as file:
            for e in evaluateSample :
                print(e, '=', self.mapLookupTest[e], file=file)

    def generateEquivalenceSet(self, arr):
        equivalence = set()
        for inst in arr:
            x, y = re.compile(' */ *').split(inst.source)
            equivalence.add(x)
            equivalence.add(y)

        arr = []
        for sent in equivalence:
            inst = ParaphraseInstance(sent, sent, 1, False)
            inst.source = inst.x + ' / ' + inst.y
            inst.x = tokenization.chinois.tokenize(inst.x)
            inst.y = tokenization.chinois.tokenize(inst.y)
            arr.append(inst)
        return arr

    def establishTrainingSet(self):
        trainingSamplePositive = []
        trainingSampleNegative = []

        for score, s in enumerate(self.map):

            score = score / 100

            arr = list(s)

            for i in range(len(arr)):
                inst = arr[i].copy()
                inst.source = inst.x + ' / ' + inst.y

                inst.x = tokenization.chinois.tokenize(inst.x)
                inst.y = tokenization.chinois.tokenize(inst.y)
                inst.score = score

                arr[i] = inst

            if score >= 0.5:
                trainingSamplePositive += arr
            else:
                trainingSampleNegative += arr

        print('tally for Positive =', len(trainingSamplePositive))
        print('tally for Negative =', len(trainingSampleNegative))

        return trainingSamplePositive + trainingSampleNegative  # + self.generateEquivalenceSet(self.map['0.50'])

    def establishEvaluateSet(self):
        trainingSamplePositive = []
        trainingSampleNegative = []

        for score, s in enumerate(self.mapTest):

            score = score / 100

            arr = list(s)

            for i in range(len(arr)):
                inst = arr[i].copy()
                inst.source = inst.x + ' / ' + inst.y

                inst.x = tokenization.chinois.tokenize(inst.x)
                inst.y = tokenization.chinois.tokenize(inst.y)
                inst.score = score

                arr[i] = inst

            if score >= 0.5:
                trainingSamplePositive += arr
            else:
                trainingSampleNegative += arr

        print('tally for Positive =', len(trainingSamplePositive))
        print('tally for Negative =', len(trainingSampleNegative))

        return trainingSamplePositive + trainingSampleNegative  # + self.generateEquivalenceSet(self.map['0.50'])

    def establishPredictSet(self):
        evaluateSample = []
#         equivalence = []
        for s in utility.Text(utility.corpusDirectory + 'unlabeled.txt'):
            try:
                x, y, score = match(s)
            except Exception as e:
                print(e)
                continue

            inst = ParaphraseInstance(x, y, score, False)
            inst.source = inst.x + ' / ' + inst.y

            inst.x = tokenization.chinois.tokenize(inst.x)
            inst.y = tokenization.chinois.tokenize(inst.y)

            evaluateSample += [inst]
#             if 0.4 < score < 0.6:
#                 equivalence.append(inst)

        print('tally for evaluation =', len(evaluateSample))
        return evaluateSample
#         return evaluateSample + self.generateEquivalenceSet(equivalence)

    def establishEvaluateCorpus(self):
        self.mapLookupTest = {}
        bRefresh = False
        validateSet = set()
        for s in utility.Text(self.evaluate_file):
            x, y, score = match(s)

            if score is None:
                score = '0'

            inst = ParaphraseInstance(x, y, score)
            self.mapLookupTest[inst] = normalize(score)

            if inst in self.mapLookup:
                bRefresh = True
#                 self.mapLookupTest[inst] = self.delete_from_training_corpus(inst, validateSet)
#                 print(inst, '=', self.mapLookupTest[inst])
#                 print('found in training set! the training set example will be deleted!')

                print(inst, '=', self.mapLookup[inst])
                print('found in training set! the testing set example will be deleted!')
                del self.mapLookupTest[inst]

        for inst in self.deleteSet:
            if inst in self.mapLookupTest:
                bRefresh = True
                print(inst, '\ndeleting from training set!')
                del self.mapLookupTest[inst]

        if bRefresh:
            self.validateTrainingCorpus(validateSet)
            self.saveTestSet()

    def cleanEvaluateCorpus(self):
        st = set()
        with open(utility.corpusDirectory + 'debug.txt', 'w', encoding='utf8') as file:
            for p in self.mapLookupTest:
                if len(p.x) < 3 and len(p.y) < 3:
                    print('%s / %s = %s' % (p.x, p.y, p.score), file=file)
                    st.add(p)

        for inst in st:
            del self.mapLookupTest[inst]

        if st:
            self.saveTestSet()

    def generateTestSet(self, file='_evaluate.txt'):
        evaluateSample = []

        for score, s in self.map.items():

            arr = list(s)
#             shuffle(arr)

            for e in arr:
                e.score = float(score)
            pivot = len(arr) // 11
            evaluateSample += arr[:pivot]
            self.map[score] = arr[pivot:]

        evaluateSample.sort(key=lambda x: x.score, reverse=True)

        with open(utility.corpusDirectory + 'paraphrase/' + file, mode='w', encoding='utf8') as file :
            for e in evaluateSample :
                print(e, '=', e.score, file=file)
        print('size =', len(evaluateSample))

        for score in self.map:
            print('update file', Corpus.folder + score + '.data')
            if score in self.map:
                utility.Text(Corpus.folder + score + '.data').write(self.map[score])
            else:
                utility.Text(Corpus.folder + score + '.data').clear()

    def search_and(self, score, keyword, _keyword):
        matcher = re.compile(keyword)
        if _keyword:
            _matcher = re.compile(_keyword)
        else :
            _matcher = matcher
        for inst in self.fetch(score):
            if (matcher.search(inst.x) and _matcher.search(inst.y)) :
                yield inst

#             if (matcher.search(inst.x) and _matcher.search(inst.y)) or (_matcher.search(inst.x) and matcher.search(inst.y)):
#                 yield inst

    def search_or(self, score, keyword, _keyword):
        matcher = re.compile(keyword)
        if _keyword:
            _matcher = re.compile(_keyword)
        else :
            _matcher = matcher
        for inst in self.fetch(score):
            if matcher.search(inst.x) or _matcher.search(inst.y) or _matcher.search(inst.x) or matcher.search(inst.y):
                yield inst

# to look for a string that does not contain any one of these: '((?!(?:怎样|怎么|咋|如何)).)+'
    def match_and(self, score, keyword, _keyword):
        matcher = re.compile(keyword)
        if _keyword:
            _matcher = re.compile(_keyword)
        else :
            _matcher = matcher
        for inst in self.fetch(score):
            if matcher.fullmatch(inst.x) and _matcher.fullmatch(inst.y) or _matcher.fullmatch(inst.x) and matcher.fullmatch(inst.y):
                yield inst

    def match_or(self, score, keyword, _keyword):
        matcher = re.compile(keyword)
        if _keyword:
            _matcher = re.compile(_keyword)
        else :
            _matcher = matcher
        for inst in self.fetch(score):
            if matcher.fullmatch(inst.x) or _matcher.fullmatch(inst.y) or _matcher.fullmatch(inst.x) or matcher.fullmatch(inst.y):
                yield inst

    def fetch(self, score):
        score = normalize(score)
        for inst in self.map[score]:
            yield inst

        for inst, _score in self.mapLookupTest.items():
            if _score == score:
                yield inst

    def print_inst(self, score):
        from paraphrase import bert_semantic
        arr = list(self.fetch(score))
        print('size =', len(arr))
        similarity = bert_semantic.instance.predict(arr)
        print('max =', similarity.max())

        minimum = similarity.min()
        print('min =', similarity.min())
#         similarity = numpy.clip(similarity, 0, 0.4)
#         if similarity > 0.4:
#             similarity = 0.4
        file = open(utility.corpusDirectory + 'debug.txt', 'w', encoding='utf8')
        for sent, s in zip(arr, similarity):
            if s < 0.1:
#                 print(s - minimum)
#                 print((s - minimum) * 10)
                s = 0.01 + (s - minimum) / (0.1 - minimum) * 0.09
                print('%s / %s = %f' % (sent.x, sent.y, s), file=file)


instance = Corpus()


def convert_sentence_pair(tokens_a, tokens_b, max_seq_length=None, mask=False):
    tokens = []
    segment_ids = []

    tokens.append(instance.tokenizer.vocab["[CLS]"])
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append(instance.tokenizer.vocab["[SEP]"])
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append(instance.tokenizer.vocab["[SEP]"])
    segment_ids.append(1)

    if mask:
        input_mask = [1] * len(tokens)

        if max_seq_length and len(tokens) < max_seq_length:
            zero_padding = [0] * (max_seq_length - len(tokens))
            tokens += zero_padding
            input_mask += zero_padding
            segment_ids += zero_padding
    #
            assert len(tokens) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

        return tokens, segment_ids, input_mask
    else:
    # Zero-pad up to the sequence length.
        if max_seq_length and len(tokens) < max_seq_length:
            zero_padding = [0] * (max_seq_length - len(tokens))
            tokens += zero_padding
            segment_ids += zero_padding
    #
            assert len(tokens) == max_seq_length
            assert len(segment_ids) == max_seq_length

        return tokens, segment_ids


def evaluate_testing_result(testSet, arr, file):

    paraphraseErr = 0
    paraphraseSgm = 0

    entailmentErr = 0
    entailmentSgm = 0

    analogicalErr = 0  # pertinence
    analogicalSgm = 0

    contradictErr = 0
    contradictSgm = 0

    irrelevantErr = 0
    irrelevantSgm = 0
    print('evaluate on test data set')

    assert len(testSet) == len(arr)
    if not os.path.exists(file):
        os.makedirs(file)
    file_paraphrase = open(file + 'paraphrase.txt', 'w', encoding='utf8')
    file_entailment = open(file + 'entailment.txt', 'w', encoding='utf8')
    file_analogical = open(file + 'analogical.txt', 'w', encoding='utf8')
    file_contradict = open(file + 'contradict.txt', 'w', encoding='utf8')
    file_irrelevant = open(file + 'irrelevant.txt', 'w', encoding='utf8')

    errSet = set()

    for inst, score in zip(testSet, arr):
        y_pred = score
        y_true = inst.score

        if y_true >= 0.9:
            paraphraseSgm += 1
        elif y_true >= 0.75:
            entailmentSgm += 1
        elif y_true >= 0.55:
            analogicalSgm += 1
        elif y_true >= 0.3:
            contradictSgm += 1
        else:
            irrelevantSgm += 1

        if not optimization.accuracy_numeric(y_true, y_pred):
            x, y = re.compile('\s*/\s*').split(inst.source)
            errInst = ParaphraseInstance(x, y)
            if errInst in errSet:
                drapeau = False
            else:
                errSet.add(errInst)
                drapeau = True

            if y_true >= 0.9:
                paraphraseErr += 1

                if drapeau:
                    print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_paraphrase)
            elif y_true >= 0.75:
                entailmentErr += 1
                if drapeau:
                    print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_entailment)
            elif y_true >= 0.55:
                analogicalErr += 1
                if drapeau:
                    print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_analogical)
            elif y_true >= 0.3:
                contradictErr += 1
                if drapeau:
                    print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_contradict)
            else:
                irrelevantErr += 1
                if drapeau:
                    print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_irrelevant)

    print('paraphrase Err =', paraphraseErr)
    print('paraphrase Sgm =', paraphraseSgm)
    print('paraphrase Acc =', (paraphraseSgm - paraphraseErr) / paraphraseSgm if paraphraseSgm else 1)

    print('entailment Err =', entailmentErr)
    print('entailment Sgm =', entailmentSgm)
    print('entailment Acc =', (entailmentSgm - entailmentErr) / entailmentSgm if entailmentSgm else 1)

    print('analogical Err =', analogicalErr)
    print('analogical Sgm =', analogicalSgm)
    print('analogical Acc =', (analogicalSgm - analogicalErr) / analogicalSgm if analogicalSgm else 1)

    print('contradict Err =', contradictErr)
    print('contradict Sgm =', contradictSgm)
    print('contradict Acc =', (contradictSgm - contradictErr) / contradictSgm if contradictSgm else 1)

    print('irrelevant Err =', irrelevantErr)
    print('irrelevant Sgm =', irrelevantSgm)
    print('irrelevant Acc =', (irrelevantSgm - irrelevantErr) / irrelevantSgm if irrelevantSgm else 1)

    holisticErr = paraphraseErr + entailmentErr + analogicalErr + contradictErr + irrelevantErr
    holisticSgm = irrelevantSgm + paraphraseSgm + entailmentSgm + analogicalSgm + contradictSgm

    print('holistic Err =', holisticErr)
    print('holistic Sgm =', holisticSgm)
    print('holistic Acc =', (holisticSgm - holisticErr) / holisticSgm)

    file_paraphrase.close()
    file_entailment.close()
    file_analogical.close()
    file_contradict.close()
    file_irrelevant.close()


def evaluate_predict_result(testSet, arr, file):

    paraphraseSgm = 0
    entailmentSgm = 0
    analogicalSgm = 0
    contradictSgm = 0
    irrelevantSgm = 0
    print('evaluate on predict data set')

    assert len(testSet) == len(arr)
    if not os.path.exists(file):
        os.makedirs(file)
    file_paraphrase = open(file + 'paraphrase.txt', 'w', encoding='utf8')
    file_entailment = open(file + 'entailment.txt', 'w', encoding='utf8')
    file_analogical = open(file + 'analogical.txt', 'w', encoding='utf8')
    file_contradict = open(file + 'contradict.txt', 'w', encoding='utf8')
    file_irrelevant = open(file + 'irrelevant.txt', 'w', encoding='utf8')

    errSet = set()

    for inst, score in zip(testSet, arr):
        y_pred = score
        y_true = y_pred

        if y_true >= 0.9:
            paraphraseSgm += 1
        elif y_true >= 0.75:
            entailmentSgm += 1
        elif y_true >= 0.55:
            analogicalSgm += 1
        elif y_true >= 0.3:
            contradictSgm += 1
        else:
            irrelevantSgm += 1

        x, y = re.compile('\s*/\s*').split(inst.source)
        errInst = ParaphraseInstance(x, y)
        if errInst in errSet:
            drapeau = False
        else:
            errSet.add(errInst)
            drapeau = True

        if y_true >= 0.9:
            if drapeau:
                print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_paraphrase)
        elif y_true >= 0.75:
            if drapeau:
                print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_entailment)
        elif y_true >= 0.55:
            if drapeau:
                print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_analogical)
        elif y_true >= 0.3:
            if drapeau:
                print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_contradict)
        else:
            if drapeau:
                print('%s = %4.2f\t%4.2f' % (inst.source, y_true, y_pred), file=file_irrelevant)

    print('paraphrase Sgm =', paraphraseSgm)
    print('entailment Sgm =', entailmentSgm)
    print('analogical Sgm =', analogicalSgm)
    print('contradict Sgm =', contradictSgm)
    print('irrelevant Sgm =', irrelevantSgm)

    holisticSgm = irrelevantSgm + paraphraseSgm + entailmentSgm + analogicalSgm + contradictSgm

    print('holistic Sgm =', holisticSgm)

    file_paraphrase.close()
    file_entailment.close()
    file_analogical.close()
    file_contradict.close()
    file_irrelevant.close()


class StringMatcher:

    def compile_regex(self, keyword):
        if keyword.startswith("'") and keyword.endswith("'"):
            return [keyword[1:-1] + '.*']
        keyword = re.compile('\s*,\s*').split(keyword)
        return [self.compile_single_regex(word) for word in keyword]

    def compile_single_regex(self, keyword):
        if keyword.startswith('^'):
            keyword = keyword[1:]
            return '((?!(?:' + keyword + ')).)+'
        return '.*(?:' + keyword + ').*'

    def __init__(self, x, y, op, score):
        self.x = self.compile_regex(x)
        self.y = self.compile_regex(y)

        self.op = op

        if score.startswith('0.'):
            self.score = score
        else :
            self.score = '0.' + score
        self.score = float(self.score) * 100
        self.corpus = []
        if self.op == '==':
            self.corpus = list(instance.fetch(self.score / 100))
        elif self.op == '>':
            for s in range(int(self.score) + 1, 101):
                self.corpus += list(instance.fetch(s / 100))
        elif self.op == '<':
            for s in range(1, int(self.score)):
                self.corpus += list(instance.fetch(s / 100))
        elif self.op == '>=':
            for s in range(int(self.score), 101):
                self.corpus += list(instance.fetch(s / 100))
        elif self.op == '<=':
            for s in range(1, int(self.score) + 1):
                self.corpus += list(instance.fetch(s / 100))
        else:
            for s in numpy.arange(0.01, 1, 0.01):
                self.corpus += list(instance.fetch(s / 100))

    def match_all(self, x, sent):
        for regex in x:
            if not re.compile(regex).fullmatch(sent):
                return False
        return True

    def match(self, paraphraseInstance):
        return self.match_all(self.x, paraphraseInstance.x) and self.match_all(self.y, paraphraseInstance.y) \
            or self.match_all(self.x, paraphraseInstance.y) and self.match_all(self.y, paraphraseInstance.x)

    def fetch(self):
        for inst in self.corpus:
            if self.match(inst):
                yield inst


def generate():
    file = utility.corpusDirectory + 'debug.txt'

    with open(file, 'w', encoding='utf8') as file:

        arr = [['星期1', '星期一', '周1', '周一'],
               ['星期2', '星期二', '周2', '周二'],
               ['星期3', '星期三', '周3', '周三'],
               ['星期4', '星期四', '周4', '周四'],
               ['星期5', '星期五', '周5', '周五'],
               ['星期6', '星期六', '周6', '周六'],
               ['星期天', '星期日', '周日'],
               ['星期几', '周几']]

        for i in range(len(arr)):
            si = arr[i]
            for _i in range(len(si)):
                for _j in range(_i + 1, len(si)):
                    print('%s / %s = 0.99' % (si[_i], si[_j]), file=file)
                    print('今天是%s / 今天是%s = 0.99' % (si[_i], si[_j]), file=file)
                    print('明天是%s / 明天是%s = 0.99' % (si[_i], si[_j]), file=file)
                    print('后天是%s / 后天是%s = 0.99' % (si[_i], si[_j]), file=file)

                    print('今天是%s / 明天是%s = 0.65' % (si[_i], si[_j]), file=file)
                    print('明天是%s / 后天是%s = 0.65' % (si[_i], si[_j]), file=file)
                    print('后天是%s / 今天是%s = 0.65' % (si[_i], si[_j]), file=file)

            for j in range(i + 1, len(arr)):
                sj = arr[j]

                for _i in range(len(si)):
                    for _j in range(len(sj)):
                        print('%s / %s = 0.60' % (si[_i], sj[_j]), file=file)
                        print('今天是%s / 今天是%s = 0.65' % (si[_i], sj[_j]), file=file)
                        print('明天是%s / 明天是%s = 0.65' % (si[_i], sj[_j]), file=file)
                        print('后天是%s / 后天是%s = 0.65' % (si[_i], sj[_j]), file=file)


def convert():
    file = utility.corpusDirectory + 'debug.txt'

    st = []
    for s in utility.Text(file):
        x, y, _ = match(s)
        if re.compile('不得不|必须|务必').search(x):
            st.append(x)
        if re.compile('不得不|必须|务必').search(y):
            st.append(y)

    for s in st:
        m = re.compile('(.*)(不得不|必须|务必)(.*)').fullmatch(s)

        s1 = m.group(1)
        s3 = m.group(3)

#         m = re.compile('(.+?)([,，。\.吗的么吧呢\?？]+.*)').fullmatch(s3)
#         if m :
#             m1 = m.group(1)
#             m2 = m.group(2)
#             if m1[-1] == '什' and m2[0] == '么':
#                 m1 = m1 + m2[0]
#                 m2 = m2[1:]
#
#             s3 = m1 + '不可' + m2
#         else:
#             s3 += '不可'
        print('%s / %s = 0.98' % (s, s1 + '非得' + s3))


def sample():
    file = utility.corpusDirectory + 'debug.txt'
    arr = utility.Text(file).collect()
    random.shuffle(arr)

    file = utility.corpusDirectory + '_debug.txt'

    with open(file, 'w', encoding='utf8') as file:
        for s in arr[:len(arr) // 10]:
            print(s, file=file)


# 杯子是什么垃圾 / 奶茶属于什么垃圾 = 0.6
if __name__ == '__main__':
#     generate()
#     instance.cleanEvaluateCorpus()
    sample()
#     convert()
#     instance.generateTestSet('_test.txt')
#     for i in range(95, 100):
#         for inst in list(instance.match_and(i / 100, '((?!(?:哪些|哪个)).)+', '.*(哪些|哪个).*')):
#             print(inst, '=', 0.93)

# = 0\.\d+    0.(7[01234]|[0123456]\d)
