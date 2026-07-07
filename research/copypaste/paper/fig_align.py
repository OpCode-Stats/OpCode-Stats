import subprocess, re
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

P="/home/aaslyan/openacr-mine/build/release/amc"
FUNC=re.compile(r'^[0-9a-f]+ <(.+?)>:$'); INS=re.compile(r'^\s*[0-9a-f]+:\s*(?:[0-9a-f]{2} )+\s+([a-zA-Z][\w.]*)')
out=subprocess.run(['objdump','-d','-M','intel',P],capture_output=True,text=True).stdout
F={};cur=None;ops=[]
for line in out.splitlines():
    m=FUNC.match(line.strip())
    if m:
        if cur and 18<=len(ops)<=34: F[cur]=ops
        cur=m.group(1);ops=[];continue
    mi=INS.match(line)
    if mi and cur is not None: ops.append(mi.group(1))
# NW align
def nw(a,b):
    n,m=len(a),len(b);H=[[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):H[i][0]=-i
    for j in range(1,m+1):H[0][j]=-j
    for i in range(1,n+1):
        for j in range(1,m+1): H[i][j]=max(H[i-1][j-1]+(1 if a[i-1]==b[j-1] else -1),H[i-1][j]-1,H[i][j-1]-1)
    i,j=n,m;al=[]
    while i>0 or j>0:
        if i>0 and j>0 and H[i][j]==H[i-1][j-1]+(1 if a[i-1]==b[j-1] else -1): al.append((a[i-1],b[j-1]));i-=1;j-=1
        elif i>0 and H[i][j]==H[i-1][j]-1: al.append((a[i-1],None));i-=1
        else: al.append((None,b[j-1]));j-=1
    return al[::-1]
# find a fuzzy pair: similar but not identical
names=list(F); best=None
for x in range(len(names)):
    for y in range(x+1,len(names)):
        a,b=F[names[x]],F[names[y]]
        if abs(len(a)-len(b))>6: continue
        al=nw(a,b); match=sum(1 for u,v in al if u==v and u is not None)
        frac=match/len(al)
        if 0.55<frac<0.93 and (best is None or len(al)<best[0]):
            best=(len(al),al,names[x],names[y],frac)
L,al,nx,ny,frac=best
import subprocess as sp
dn=lambda s: sp.run(['c++filt'],input=s,capture_output=True,text=True).stdout.strip()
fig,ax=plt.subplots(figsize=(7.0,1.9))
for c,(u,v) in enumerate(al):
    same=(u==v and u is not None); col='#bfe3bf' if same else '#f6c0c0'
    for row,tok in [(1,u),(0,v)]:
        ax.add_patch(Rectangle((c,row),1,1,facecolor=col,edgecolor='white'))
        ax.text(c+0.5,row+0.5,tok if tok else '–',ha='center',va='center',fontsize=5.5,rotation=90)
ax.add_patch(Rectangle((0,-0.9),L,0.32,facecolor='none',edgecolor='none'))
# bracket labels
ax.text(L/2,2.35,f"two near-duplicate amc functions  ({int(frac*100)}% conserved skeleton, rest = holes)",ha='center',fontsize=8)
ax.text(-0.4,1.5,"A",ha='center',va='center',fontsize=8,fontweight='bold')
ax.text(-0.4,0.5,"B",ha='center',va='center',fontsize=8,fontweight='bold')
ax.add_patch(Rectangle((0,2.0),3.0,0.0,facecolor='#bfe3bf'))
ax.text(0.05,-0.55,"green = MATCH (skeleton, identical incl. operands)   red = DIFFER (hole -> parameter)",fontsize=6.5)
ax.set_xlim(-0.8,L); ax.set_ylim(-0.9,2.6); ax.axis('off')
plt.savefig('fig_alignment.pdf',bbox_inches='tight'); print(f"fig_alignment.pdf  pair=({dn(nx)[:30]} , {dn(ny)[:30]})  cols={L} frac={frac:.2f}")
