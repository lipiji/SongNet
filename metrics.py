import os
import sys
import numpy as np
from pypinyin import Style, lazy_pinyin

from data import PUNCS

yunjiaos = {
            "0":["a", "ia", "ua", "va", "üa"],
            "1":["e", "o", "uo", "ie", "ue", "üe", "ve"],
            "2":["u"],
            "3":["i", "ü", "v"],
            "4":["ai", "uai"],
            "5":["ao", "iao"],
            "6":["ou", "iu", "iou"],
            "7":["an", "ian", "uan", "üan", "van"],
            "8":["en", "in", "un", "ün", "vn"],
            "9":["ang", "iang", "uang"],
            "10":["eng", "ing", "ueng", "ong", "iong"],
            "11":["er"],
            "12":["ei", "ui", "uei", "vei"],
           }

yun2id = {}
for yid, yws in yunjiaos.items():
    for w in yws:
        yun2id[w] = yid

def eval_tpl(sents1, sents2):
    n = 0.
    if len(sents1) > len(sents2):
        sents1 = sents1[:len(sents2)]
    for i, x in enumerate(sents1):
        y = sents2[i]
        if len(x) != len(y):
            continue
        px, py = [], []
        for w in x:
            if w in PUNCS:
                px.append(w)
        for w in y:
            if w in PUNCS:
                py.append(w)
        if px == py:
            n += 1
    p = n / len(sents2)
    r = n / len(sents1)
    f = 2 * p * r / (p + r + 1e-16)

    return p, r, f, n, len(sents1), len(sents2)


def rhythm_labellig(sents):
    rhys = []
    for sent in sents:
        w = sent[-1]
        if w in PUNCS and len(sent) > 1:
            w = sent[-2]
        yunmu = lazy_pinyin(w, style=Style.FINALS)
        rhys.append(yunmu[0])
    assert len(rhys) == len(sents)
    rhy_map = {}
    for i, r in enumerate(rhys):
        if r in yun2id:
            rid = yun2id[r]
            if rid in rhy_map:
                rhy_map[rid] += [i]
            else:
                rhy_map[rid] = [i]
        else:
            pass
    max_len_yuns = -1
    max_rid = ""
    for rid, yuns in rhy_map.items():
        if len(yuns) > max_len_yuns:
            max_len_yuns = len(yuns)
            max_rid = rid
    res = []
    for i in range(len(sents)):
        if max_rid in rhy_map and i in rhy_map[max_rid]:
            res.append(1)
        else:
            res.append(-1)
    return res

def eval_rhythm(sents1, sents2):
    n = 0.
    if len(sents1) > len(sents2):
        sents1 = sents1[:len(sents2)]
    rhys1 = rhythm_labellig(sents1)
    rhys2 = rhythm_labellig(sents2)
    
    n1, n2 = 0., 0.
    for v in rhys1:
        if v == 1:
            n1 += 1
    for v in rhys2:
        if v == 1:
            n2 += 1
    for i, v1 in enumerate(rhys1):
        v2 = rhys2[i]
        if v1 == 1 and v1 == v2:
            n += 1
    p = n / (n2 + 1e-16)
    r = n / (n1 + 1e-16)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, f1, n, n1, n2

def eval_endings(sents1, sents2):
    n = 0.
    if len(sents1) > len(sents2):
        sents1 = sents1[:len(sents2)]
   
    sents0 = []
    for si, sent1 in enumerate(sents1):
        sent2 = sents2[si]
        if len(sent2) <= len(sent1):
            sents0.append(sent2)
        else:
            sents0.append(sent2[:len(sent1) - 1] + sent1[-1])

    sent = "</s>".join(sents0)
    return sent


def eval(res_file, fid):
    docs = []
    with open(res_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fs = line.split("\t")
            if len(fs) != 2:
                print("error", line)
                continue
            x, y = fs
            docs.append((x, y))


    print(len(docs))

    ugrams_ = []
    bigrams_ = []
    p_, r_, f1_ = 0., 0., 0.
    n0_, n1_, n2_ = 0., 0., 0.

    p__, r__, f1__ = 0., 0., 0.
    n0__, n1__, n2__ = 0., 0., 0.
    d1_, d2_ = 0., 0.
    d4ends = []

    for x, y in docs:
        topic, content = x.split("<s2>")
        author, topic = topic.split("<s1>")
        sents1 = content.split("</s>")
        y = y.replace("<bos>", "")
        sents2 = y.split("</s>")
        sents1_ = []
        for sent in sents1:
            sent = sent.strip()
            if sent:
                sents1_.append(sent)
        sents1 = sents1_
        sents2_ = []
        for sent in sents2:
            sent = sent.strip()
            if sent:
                sents2_.append(sent)
        sents2 = sents2_

        p, r, f1, n0, n1, n2 = eval_tpl(sents1, sents2)
        p_ += p
        r_ += r
        f1_ += f1
        n0_ += n0
        n1_ += n1
        n2_ += n2

        ugrams = [w for w in ''.join(sents2)]
        bigrams = []
        for bi in range(len(ugrams) - 1):
            bigrams.append(ugrams[bi] + ugrams[bi+1])
        d1_ += len(set(ugrams)) / float(len(ugrams))
        d2_ += len(set(bigrams)) / float(len(bigrams))
        ugrams_ += ugrams
        bigrams_ += bigrams

        p, r, f1, n0, n1, n2 = eval_rhythm(sents1, sents2)
        p__ += p
        r__ += r
        f1__ += f1
        n0__ += n0
        n1__ += n1
        n2__ += n2

        d4end = eval_endings(sents1, sents2)
        d4ends.append(author + "<s1>" + topic + "<s2>" + d4end)

    tpl_macro_p = p_ / len(docs)
    tpl_macro_r = r_ / len(docs)
    tpl_macro_f1 = 2 * tpl_macro_p * tpl_macro_r / (tpl_macro_p + tpl_macro_r)
    tpl_micro_p = n0_ / n2_
    tpl_micro_r = n0_ / n1_
    tpl_micro_f1 = 2 * tpl_micro_p * tpl_micro_r / (tpl_micro_p + tpl_micro_r)
    
    rhy_macro_p = p__ / len(docs)
    rhy_macro_r = r__ / len(docs)
    rhy_macro_f1 = 2 * rhy_macro_p * rhy_macro_r / (rhy_macro_p + rhy_macro_r)
    rhy_micro_p = n0__ / n2__
    rhy_micro_r = n0__ / n1__
    rhy_micro_f1 = 2 * rhy_micro_p * rhy_micro_r / (rhy_micro_p + rhy_micro_r)
    

    macro_dist1 = d1_ / len(docs)
    macro_dist2 = d2_ / len(docs)
    micro_dist1 = len(set(ugrams_)) / float(len(ugrams_))
    micro_dist2 = len(set(bigrams_)) / float(len(bigrams_))

    with open("./results_4ending/res4end" + str(fid) + ".txt", "w") as fo:
        for line in d4ends:
            fo.write(line + "\n")
    return tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2

tpl_macro_f1_, tpl_micro_f1_, rhy_macro_f1_, rhy_micro_f1_,  \
macro_dist1_, micro_dist1_, macro_dist2_, micro_dist2_ = [], [], [], [], [], [], [], []
abalation = "top-32"
for i in range(5):
    f_name = "./results/"+abalation+"/out" +str(i+1)+".txt"
    if not os.path.exists(f_name):
        continue
    tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2 = eval(f_name, i + 1)
    print(tpl_macro_f1, tpl_micro_f1, rhy_macro_f1, rhy_micro_f1, macro_dist1, micro_dist1, macro_dist2, micro_dist2)
    tpl_macro_f1_.append(tpl_macro_f1)
    tpl_micro_f1_.append(tpl_micro_f1)
    rhy_macro_f1_.append(rhy_macro_f1)
    rhy_micro_f1_.append(rhy_micro_f1)
    macro_dist1_.append(macro_dist1)
    micro_dist1_.append(micro_dist1)
    macro_dist2_.append(macro_dist2)
    micro_dist2_.append(micro_dist2)

print()
print("tpl_macro_f1", np.mean(tpl_macro_f1_), np.std(tpl_macro_f1_, ddof=1))
print("tpl_micro_f1", np.mean(tpl_micro_f1_), np.std(tpl_micro_f1_, ddof=1))
print("rhy_macro_f1", np.mean(rhy_macro_f1_), np.std(rhy_macro_f1_, ddof=1))
print("rhy_micro_f1", np.mean(rhy_micro_f1_), np.std(rhy_micro_f1_, ddof=1))
print("macro_dist1", np.mean(macro_dist1_), np.std(macro_dist1_, ddof=1))
print("micro_dist1", np.mean(micro_dist1_), np.std(micro_dist1_, ddof=1))
print("macro_dist2", np.mean(macro_dist2_), np.std(macro_dist2_, ddof=1))
print("micro_dist2", np.mean(micro_dist2_), np.std(micro_dist2_, ddof=1))





