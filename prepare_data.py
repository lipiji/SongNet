import sys, re
from collections import Counter
import random
cnt = Counter()

f_ci = "./data/ci.txt"
f_cipai = "./data/cipai.txt"
cipai = Counter()
with open(f_cipai) as f:
    for line in f:
        line = line.strip()
        fs = line.split()
        cipai.update(fs)

cipai = cipai.keys()

docs = {}
with open(f_ci) as f:
    for line in f:
        line = line.strip()
        fs = line.split("<s1>")
        author = fs[0]
        topic, content = fs[1].split("<s2>")
        if "・" in topic:
            t1, t2 = topic.split("・")
            if t1 == t2:
                topic = t1
            else:
                if t1 in cipai:
                    topic = t1
                elif t2 in cipai:
                    topic = t2
                else:
                    topic = t1
        content = content.replace("、", "，")
        sents = content.split("</s>")
        ws = [w for w in author + topic + ''.join(sents)]
        cnt.update(ws)
        if topic not in docs:
            docs[topic] = []
        docs[topic].append(author + "<s1>" + topic + "<s2>" + '</s>'.join(sents))


topics = list(docs.keys())

print(len(topics))
random.shuffle(topics)

topics_train = topics[:len(topics)-50]
topics_dev_test = topics[-50:]
topics_dev = topics_dev_test[:25]
topics_test = topics_dev_test[-25:]

docs_train = []
docs_dev = []
docs_test = []

for t in topics_train:
    docs_train.extend(docs[t])

for t in topics_dev:
    docs_dev.extend(docs[t])

for t in topics_test:
    docs_test.extend(docs[t])

random.shuffle(docs_train)
random.shuffle(docs_dev)
random.shuffle(docs_test)

print(len(docs_train), len(docs_dev), len(docs_test))
train_cps = []
dev_cps = []
test_cps = []


with open('./data/train.txt', 'w', encoding ='utf8') as f:
    for x in docs_train:
        s = x.split("<s2>")[0]
        train_cps.append(s.split("<s1>")[1])
        f.write(x + '\n')
    print(len(set(train_cps)))
with open('./data/dev.txt', 'w', encoding ='utf8') as f:
    for x in docs_dev:
        s = x.split("<s2>")[0]
        dev_cps.append(s.split("<s1>")[1])
        f.write(x + '\n')
    print(len(set(dev_cps)))
with open('./data/test.txt', 'w', encoding ='utf8') as f:
    for x in docs_test:
        s = x.split("<s2>")[0]
        test_cps.append(s.split("<s1>")[1])
        f.write(x + '\n')
    print(len(set(test_cps)))

print("vocab")
with open('./data/vocab.txt', 'w', encoding ='utf8') as f:
    for x, y in cnt.most_common():
        f.write(x + '\t' + str(y) + '\n')
print("done")
