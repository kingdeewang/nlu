'''

@author: Cosmos
'''

import re
from builtins import range
import os
from gc import collect
from util import utility
workingDirectory = os.path.abspath(os.path.join(os.getcwd(), "../../"))
modelsDirectory = workingDirectory + "/models/"
corpusDirectory = workingDirectory + "/corpus/"

charMap = {}

for line in utility.Text(modelsDirectory + "simplify.txt"):
    res = re.compile("(.+)\\s*=>\\s*(.)").fullmatch(line).groups()
    y = res[1]
    for ch in res[0]:
        if not ch.isspace():
            charMap[ch] = y


def simplifyString(seg):
    if seg == None:
        return None

    last = '\0'
    for i in range(len(seg)):
        ch = seg[i]
        if not ch in charMap and ch != last:
            last = ch
            continue

        s = [seg[0: i]]
        for i in range(i, len(seg)):
            ch = seg[i]
            if ch in charMap:
                ch = charMap[ch]

            if ch != last:
                last = ch
            else:
                continue

            s.append(last)

        return ''.join(s)

    return seg


def simplifyStrings(segs):
    for i in range(len(segs)):
        segs[i] = simplifyString(segs[i])
    return segs


sEnglishPunctuation = ",.:;!?()[]{}'\"=<>"
sChinesePunctuation = "，。：；！？（）「」『』【】～‘’′”“《》、…．·"
sPunctuation = sEnglishPunctuation + sChinesePunctuation


def convertFromSegmentation(arr):
    s = ""
    for i in range(len(arr) - 1):
        s += arr[i]

        if arr[i][-1] in sPunctuation or arr[i + 1][0] in sPunctuation:
            continue
        s += " "

    s += arr[-1];
    return str;


if __name__ == '__main__':
    s = ""
    exp = ""
    m = re.compile(exp).match(s)
    if m:
        print(m.groups())
