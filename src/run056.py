import os,sys,json,time,gzip,io,itertools,numpy as np,pandas as pd,torch,urllib.request
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
torch.set_num_threads(2); D='data/'; R='results/results056.json'
def log(*a): print(time.strftime('%H:%M:%S'),*a,flush=True)
DS={'BindingDB':'BindingDB','BioSNAP':'BIOSNAP_full_data'}
def load(ds):
    out={}
    for s in ('train','val','test'):
        d=pd.read_csv(f'{D}{DS[ds]}/{s}.csv'); d=d[['SMILES','Target Sequence','Label']+(['DrugBank ID','Gene'] if ds=='BioSNAP' else [])]; out[s]=d
    return out
# ---------- features
def ecfp(smi):
    from rdkit import Chem; from rdkit.Chem import AllChem; from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
    m=Chem.MolFromSmiles(smi) if isinstance(smi,str) else smi
    if m is None: return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,2048),dtype=np.float32)
def kmer2(seq):
    AA='ACDEFGHIKLMNPQRSTVWY'; ix={a:i for i,a in enumerate(AA)}; v=np.zeros(400,np.float32)
    for a,b in zip(seq,seq[1:]):
        if a in ix and b in ix: v[ix[a]*20+ix[b]]+=1
    return v/max(v.sum(),1)
_esm=None
def esm_embed(seqs):
    global _esm
    from transformers import AutoTokenizer,AutoModel
    if _esm is None:
        _esm=(AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'),AutoModel.from_pretrained('facebook/esm2_t6_8M_UR50D').eval())
    tok,m=_esm; out=[]
    with torch.no_grad():
        for i,s in enumerate(seqs):
            t=tok(s[:1022],return_tensors='pt'); h=m(**t).last_hidden_state[0,1:-1]; out.append(h.mean(0).numpy())
            if i%500==0: log('esm',i,len(seqs))
    return np.array(out,dtype=np.float32)
def features(frames):
    """unique drugs/proteins over all frames -> caches"""
    cp=D+'feat_cache.npz'
    drugs=sorted({s for f in frames for d in f.values() for s in d.SMILES}); prots=sorted({s for f in frames for d in f.values() for s in d['Target Sequence']})
    if os.path.exists(cp):
        z=np.load(cp,allow_pickle=True)
        if list(z['drugs'])==drugs and list(z['prots'])==prots: return {k:z[k] for k in z.files}
    E=[ecfp(s) for s in drugs]; bad=[i for i,e in enumerate(E) if e is None]; E=np.array([e if e is not None else np.zeros(2048,np.float32) for e in E])
    K=np.array([kmer2(s) for s in prots]); P=esm_embed(prots)
    np.savez(cp,drugs=np.array(drugs,dtype=object),prots=np.array(prots,dtype=object),E=E,K=K,P=P,bad=np.array(bad)); return {k:v for k,v in np.load(cp,allow_pickle=True).items()}
# ---------- models
def mlp_fit_predict(Xtr,ytr,Xte,seed,epochs=15):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    mu=Xtr.mean(0,dtype=np.float32);sd=(Xtr.std(0)+1e-6).astype(np.float32); Xt=torch.from_numpy((Xtr-mu)/sd);yt=torch.tensor(ytr,dtype=torch.float32);Xe=torch.from_numpy((Xte-mu)/sd)
    m=torch.nn.Sequential(torch.nn.Linear(Xt.shape[1],512),torch.nn.ReLU(),torch.nn.Dropout(0.2),torch.nn.Linear(512,512),torch.nn.ReLU(),torch.nn.Dropout(0.2),torch.nn.Linear(512,1))
    opt=torch.optim.Adam(m.parameters(),1e-3); lf=torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        m.train(); perm=torch.randperm(len(Xt),generator=g)
        for i in range(0,len(Xt),256):
            b=perm[i:i+256]; loss=lf(m(Xt[b]).squeeze(1),yt[b]); opt.zero_grad();loss.backward();opt.step()
    m.eval()
    import gc;gc.collect()
    with torch.no_grad(): return torch.sigmoid(m(Xe).squeeze(1)).numpy()
def rf_fit_predict(Xtr,ytr,Xte): return RF(300,max_features='sqrt',n_jobs=2,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1]
def met(y,p): return dict(auroc=float(roc_auc_score(y,p)),auprc=float(average_precision_score(y,p)),acc=float(accuracy_score(y,p>0.5)),n=int(len(y)),pos=float(np.mean(y)))
def agg(ms): 
    a=np.array([m['auroc'] for m in ms]);return dict(auroc_mean=float(a.mean()),auroc_sd=float(a.std(ddof=1)) if len(a)>1 else None,auprc_mean=float(np.mean([m['auprc'] for m in ms])),acc_mean=float(np.mean([m['acc'] for m in ms])),n=ms[0]['n'],pos=ms[0]['pos'])
class Pairs:
    def __init__(s,df,F,extra=None):
        s.dix=np.searchsorted(F['drugs'].astype(str),df.SMILES.values);s.pix=np.searchsorted(F['prots'].astype(str),df['Target Sequence'].values);s.y=df.Label.values.astype(int);s.df=df
def X(F,pr,kind='esm',kg=None,idx=None):
    idx=np.arange(len(pr.y)) if idx is None else idx
    parts=[F['E'][pr.dix[idx]], F['P' if kind=='esm' else 'K'][pr.pix[idx]]]
    if kg is not None: parts+= [kg[0][idx],kg[1][idx]]
    return np.hstack(parts)
def seeds_mlp(Xtr,ytr,Xte,yte,n=3): 
    return agg([met(yte,mlp_fit_predict(Xtr,ytr,Xte,s)) for s in range(n)])
def done(res,k): return k in res
def main():
    res=json.load(open(R)) if os.path.exists(R) else {}
    sv=lambda: json.dump(res,open(R,'w'),indent=1,default=float)
    data={ds:load(ds) for ds in DS}; F=features([data[d] for d in data]); log('features ready',F['E'].shape,F['P'].shape)
    PR={ds:{s:Pairs(data[ds][s],F) for s in data[ds]} for ds in data}
    if 'D1' not in res:
        res['D1']={'unparsable_smiles_unique':int(len(F['bad']))}
        for ds in DS:
            tr,te=PR[ds]['train'],PR[ds]['test']
            res['D1'][ds]={s:dict(n=int(len(PR[ds][s].y)),pos_rate=float(PR[ds][s].y.mean())) for s in PR[ds]}
            res['D1'][ds]['test_pairs_drug_seen_in_train']=float(np.isin(te.dix,tr.dix).mean());res['D1'][ds]['test_pairs_prot_seen_in_train']=float(np.isin(te.pix,tr.pix).mean())
            res['D1'][ds]['test_pairs_both_seen']=float((np.isin(te.dix,tr.dix)&np.isin(te.pix,tr.pix)).mean())
        sv();log('D1',res['D1'])
    # D2/D3/D4/D7/D10 on official split
    for ds in DS:
        k=f'off_{ds}'
        if k in res: continue
        tr,va,te=PR[ds]['train'],PR[ds]['val'],PR[ds]['test']; r={}
        Xtr,Xte=X(F,tr),X(F,te); r['M1']=seeds_mlp(Xtr,tr.y,Xte,te.y)
        r['B3_kmer']=seeds_mlp(X(F,tr,'kmer'),tr.y,X(F,te,'kmer'),te.y)
        r['B4_rf']=met(te.y,rf_fit_predict(Xtr,tr.y,Xte))
        r['drug_only_rf']=met(te.y,rf_fit_predict(F['E'][tr.dix],tr.y,F['E'][te.dix]))
        r['prot_only_rf']=met(te.y,rf_fit_predict(F['P'][tr.pix],tr.y,F['P'][te.pix]))
        # D10a permutation, D10b entity memorisation
        rs=np.random.RandomState(0); r['perm_null']=agg([met(te.y,mlp_fit_predict(Xtr,rs.permutation(tr.y),Xte,s)) for s in range(3)])
        g=tr.y.mean(); dm=pd.Series(tr.y).groupby(tr.dix).mean(); pm=pd.Series(tr.y).groupby(tr.pix).mean()
        sc=(pd.Series(te.dix).map(dm).fillna(g).values+pd.Series(te.pix).map(pm).fillna(g).values)/2; r['entity_memorisation']=met(te.y,sc)
        # D7: thresholds
        p=np.mean([mlp_fit_predict(Xtr,tr.y,np.vstack([X(F,va),Xte]),s) for s in range(3)],0); pv,pt=p[:len(va.y)],p[len(va.y):]
        ths=np.linspace(0.05,0.95,19); bt=ths[int(np.argmax([accuracy_score(va.y,pv>t) for t in ths]))]
        r['D7']=dict(auprc_M1_ensemble=float(average_precision_score(te.y,pt)),acc_at_05=float(accuracy_score(te.y,pt>0.5)),val_best_threshold=float(bt),acc_test_at_val_threshold=float(accuracy_score(te.y,pt>bt)),constant_auprc=float(te.y.mean()))
        res[k]=r; sv(); log(k,json.dumps({a:(b['auroc_mean'] if 'auroc_mean' in b else b.get('auroc')) for a,b in r.items() if a!='D7'}))
    # D5/D6 cold splits on pooled pairs
    for ds in DS:
        k=f'cold_{ds}'
        if k in res: continue
        al=pd.concat([data[ds][s] for s in ('train','val','test')],ignore_index=True); pr=Pairs(al,F); rs=np.random.RandomState(0)
        ud=np.unique(pr.dix);up=np.unique(pr.pix); hd=set(rs.permutation(ud)[:int(0.2*len(ud))]);hp=set(rs.permutation(up)[:int(0.2*len(up))])
        din=np.array([d in hd for d in pr.dix]);pin=np.array([p in hp for p in pr.pix])
        sp_={'cold_drug':(~din,din),'cold_protein':(~pin,pin),'cold_both':(~din&~pin,din&pin)}; r={}
        for nm,(trm,tem) in sp_.items():
            tri,tei=np.where(trm)[0],np.where(tem)[0]; r[nm]=dict(n_train=int(len(tri)),n_test=int(len(tei)),M1=seeds_mlp(X(F,pr,idx=tri),pr.y[tri],X(F,pr,idx=tei),pr.y[tei]))
            if nm!='cold_drug': r[nm]['B3_kmer']=seeds_mlp(X(F,pr,'kmer',idx=tri),pr.y[tri],X(F,pr,'kmer',idx=tei),pr.y[tei])
        res[k]=r; sv(); log(k,json.dumps({a:(b['M1']['auroc_mean'],b.get('B3_kmer',{}).get('auroc_mean')) for a,b in r.items()}))
    # D8 KG (BioSNAP)
    if 'D8' not in res:
        res['D8']=kg_test(data,F); sv(); log('D8',json.dumps(res['D8'])[:400])
    if 'D9' not in res:
        res['D9']=nr_test(data,F,PR); sv(); log('D9',json.dumps(res['D9'])[:400])
    log('DONE')
def get(url,fn):
    if not os.path.exists(fn): urllib.request.urlretrieve(url,fn)
    return fn
def kg_test(data,F):
    H='https://github.com/hetio/hetionet/raw/main/hetnet/tsv/'
    ed=pd.read_csv(get(H+'hetionet-v1.0-edges.sif.gz',D+'het_edges.sif.gz'),sep='\t',compression='gzip'); nd=pd.read_csv(get(H+'hetionet-v1.0-nodes.tsv',D+'het_nodes.tsv'),sep='\t')
    sym={n:i for i,n,k in zip(nd['id'],nd['name'],nd['kind']) if k=='Gene'}
    drug_types={'CcSE','CtD','CpD'}; gene_types={'DaG','DuG','DdG','GpPW','GpBP','GcG','GiG'}
    ed=ed[ed.metaedge.isin(drug_types|gene_types)]
    def mat(df,rows,cols_by_type,rowside):
        ri={r:i for i,r in enumerate(rows)}; blocks=[]
        for t in cols_by_type:
            e=df[df.metaedge==t]; 
            a,b=(e.source,e.target) if rowside=='source' else (e.target,e.source)
            if t in ('GcG','GiG'):  # symmetric, use both directions
                a=pd.concat([e.source,e.target]);b=pd.concat([e.target,e.source]);keep=a.isin(ri)
            else: keep=a.isin(ri)
            a,b=a[keep],b[keep]; cj={c:i for i,c in enumerate(sorted(set(b)))}
            blocks.append(sp.csr_matrix((np.ones(len(a)),([ri[x] for x in a],[cj[x] for x in b])),shape=(len(rows),len(cj))))
        return sp.hstack(blocks).tocsr()
    comps=sorted(set(ed[ed.metaedge.isin(drug_types)].source)|set(ed[ed.metaedge.isin(['CtD','CpD'])].source)); gens=sorted(set(nd[nd.kind=='Gene']['id']))
    Md=mat(ed[ed.metaedge.isin(drug_types)],comps,['CcSE','CtD','CpD'],'source'); 
    # gene-side: DaG etc have Disease as source for DdG? gene is source for DaG/DuG/DdG? ensure by testing role
    ge=ed[ed.metaedge.isin(gene_types)]
    gid=set(gens); 
    def side(df): # orient so gene is first
        sw=~df.source.isin(gid); return pd.DataFrame({'source':np.where(sw,df.target,df.source),'target':np.where(sw,df.source,df.target),'metaedge':df.metaedge})
    ge=side(ge); Mg=mat(ge,gens,sorted(gene_types),'source')
    Sd=TruncatedSVD(64,random_state=0).fit_transform(Md).astype(np.float32); Sg=TruncatedSVD(64,random_state=0).fit_transform(Mg).astype(np.float32)
    cidx={c.replace('Compound::',''):i for i,c in enumerate(comps)}; gidx={g:i for i,g in enumerate(gens)}
    al=pd.concat([data['BioSNAP'][s] for s in ('train','val','test')],ignore_index=True); pr=Pairs(al,F)
    dkg=np.array([Sd[cidx[x]] if x in cidx else np.zeros(64,np.float32) for x in al['DrugBank ID']]);gkg=np.array([Sg[gidx[sym[x]]] if x in sym and sym[x] in gidx else np.zeros(64,np.float32) for x in al['Gene']])
    cov=dict(drug_rows_with_kg=float(np.mean([x in cidx for x in al['DrugBank ID']])),gene_rows_with_kg=float(np.mean([x in sym and sym[x] in gidx for x in al['Gene']])))
    rs=np.random.RandomState(0); ud=np.unique(pr.dix);up=np.unique(pr.pix); hd=set(rs.permutation(ud)[:int(0.2*len(ud))]);hp=set(rs.permutation(up)[:int(0.2*len(up))])
    din=np.array([d in hd for d in pr.dix]);pin=np.array([p in hp for p in pr.pix]); tri=np.where(~din&~pin)[0];tei=np.where(din&pin)[0]
    off=len(data['BioSNAP']['train']);offv=len(data['BioSNAP']['val']); otr=np.arange(off); ote=np.arange(off+offv,len(al))
    out=dict(coverage=cov)
    for nm,(a,b) in {'cold_both':(tri,tei),'official':(otr,ote)}.items():
        out[nm]=dict(M1=seeds_mlp(X(F,pr,idx=a),pr.y[a],X(F,pr,idx=b),pr.y[b]),M1_KG=seeds_mlp(X(F,pr,kg=(dkg,gkg),idx=a),pr.y[a],X(F,pr,kg=(dkg,gkg),idx=b),pr.y[b]))
    return out
def nr_test(data,F,PR):
    import urllib.request
    base='http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/nr_admat_dgc.txt'; fn=get(base,D+'nr_admat.txt')
    A=pd.read_csv(fn,sep='\t',index_col=0); prots=list(A.index); drugs=list(A.columns)
    def kegg(u):
        try: return urllib.request.urlopen(u,timeout=20).read().decode()
        except Exception: return ''
    from rdkit import Chem
    smi={};seq={}
    for d in drugs:
        mb=kegg(f'https://rest.kegg.jp/get/{d}/mol')
        m=Chem.MolFromMolBlock(mb) if mb.strip() else None
        if m is not None: smi[d]=m
    for p in prots:
        t=kegg(f'https://rest.kegg.jp/get/{p[:3]}:{p[3:]}/aaseq')
        if t.startswith('>'): seq[p]=''.join(t.split('\n')[1:])
    cov=dict(drugs_total=len(drugs),drugs_ok=len(smi),prots_total=len(prots),prots_ok=len(seq))
    if len(smi)<0.8*len(drugs) or len(seq)<0.8*len(prots): return dict(status='NOT TESTED: coverage < 80%',coverage=cov)
    tr=pd.concat([PR['BindingDB']['train'].df,PR['BindingDB']['val'].df]); 
    trp=Pairs(tr,F); Xtr=X(F,trp)
    Dn=np.array([ecfp(smi[d]) for d in smi]);Pn=esm_embed([seq[p] for p in seq]); dl=list(smi);pl=list(seq)
    rows=[(i,j) for i,d in enumerate(dl) for j,p in enumerate(pl)]; Xte=np.array([np.hstack([Dn[i],Pn[j]]) for i,j in rows],dtype=np.float32); yte=np.array([A.loc[pl[j],dl[i]] for i,j in rows]).astype(int)
    pm=np.mean([mlp_fit_predict(Xtr,trp.y,Xte,s) for s in range(3)],0)
    pp=rf_fit_predict(F['P'][trp.pix],trp.y,Pn); pps=np.array([pp[j] for i,j in rows])
    return dict(status='tested',coverage=cov,positives=int(yte.sum()),pairs=int(len(yte)),M1=met(yte,pm),protein_only_rf=met(yte,pps))
if __name__=='__main__': main()
