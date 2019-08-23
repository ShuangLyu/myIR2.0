from getdoc2vec import GetDoc2vec
from randomcube import RandomCube
from KNNGraph import KNNG
import pandas as pd
import read_write
import time

def get_nbit(docvec,kbest):
    l = docvec.shape[0]/kbest
    nbit = 0
    while True:
        if 2**nbit >= l:
            return nbit-1
        else:
            nbit += 1

def rukou(k_best,qdoc,doc):
    queryvec = GetDoc2vec()
    newdocvec = queryvec.tomatrix(qdoc)
    q = newdocvec[-1:]
    doc2vec = newdocvec[:-1]
    n_bit = get_nbit(doc2vec, k_best)
    rdc = RandomCube(n_bitcount = n_bit, k_dim = doc2vec.shape[1])
    rdc.inbucket(doc2vec)
    start_docsid = rdc.querydoc(q)
    if start_docsid is not False:
        knng = KNNG(n_docs = doc2vec.shape[0], k_best = k_best)
        similarid = knng.search(q, start_docsid, doc2vec)
        print(doc[similarid])
    else:
        print("找不到相似内容")

if __name__ == "__main__":
    stime = time.time()
    q = ['反映仪征化纤设备工程公司改制时，其被待岗处理问题。']
    rukou(k_best =20,qdoc = q,\
          doc = pd.read_csv('D:/xftest/gkxx.csv', ',', encoding = 'ansi', header = 0)['GKXX'].astype(str))
    print(time.time()-stime)

