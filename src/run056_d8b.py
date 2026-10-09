"""D8b ADDENDUM (disclosed after seeing D8): the first D8 run had 0% gene coverage because BioSNAP 'Gene' holds UniProt accessions while Hetionet nodes are gene symbols (my mapping bug). Fix: map accession -> primary gene symbol with the UniProt REST API (rest.uniprot.org, fetched at run time, saved in data/acc2sym.json), then rerun the SAME D8 code and gate. Original D8 result is kept in results056.json as D8 (gene coverage 0)."""
import json,os,urllib.request,pandas as pd,sys
sys.path.insert(0,'src'); import run056 as r
data={ds:r.load(ds) for ds in r.DS}; F=r.features([data[d] for d in data])
acc=sorted({a for s in data['BioSNAP'] for a in data['BioSNAP'][s].Gene}); fn='data/acc2sym.json'
if not os.path.exists(fn):
    m={}
    for i in range(0,len(acc),100):
        t=urllib.request.urlopen('https://rest.uniprot.org/uniprotkb/accessions?accessions='+','.join(acc[i:i+100])+'&fields=accession,gene_primary&format=tsv',timeout=60).read().decode().split('\n')[1:]
        for l in t:
            p=l.split('\t')
            if len(p)>1 and p[1]: m[p[0]]=p[1]
    json.dump(m,open(fn,'w'))
m=json.load(open(fn)); print('mapped',len(m),'of',len(acc),flush=True)
R=json.load(open(r.R)); R['D8b']=r.kg_test(data,F,m); R['D8b']['accessions_mapped']=[len(m),len(acc)]; json.dump(R,open(r.R,'w'),indent=1,default=float); print(json.dumps(R['D8b']))
