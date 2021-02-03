#!/usr/bin/env python3
# coding: utf-8

from kgqa.question_classifier import *
from kgqa.question_parser import *
from kgqa.answer_search import *

'''问答类'''
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return ''
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        print("------------------")
        print(final_answers)
        if not final_answers:
            return ''
        else:
            return '\n'.join(final_answers)

if __name__ == '__main__':
    handler = ChatBotGraph()
    while 1:
        question = input('用户:')
        answer = handler.chat_main(question)
        print('小勇:', answer)

