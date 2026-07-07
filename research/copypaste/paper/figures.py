import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'figure.dpi':150,'savefig.bbox':'tight'})

# Fig 1: the "why bother" comparison
fig,ax=plt.subplots(figsize=(3.4,2.2))
labs=['MachOutliner\n(-O3)','MachOutliner\n(-Oz)','Our merge\n(-O3)','Merge+\nOutliner']
vals=[3.87,0.0,22.68,26.45]
cols=['#bbbbbb','#bbbbbb','#1f77b4','#2ca02c']
b=ax.bar(labs,vals,color=cols)
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.4,f'{v:.1f}%',ha='center',fontsize=8)
ax.set_ylabel('% of .text removed'); ax.set_ylim(0,30)
ax.set_title('Recoverable near-duplicate code (one family, real code)',fontsize=8.5)
plt.savefig('fig_compare.pdf'); plt.close()

# Fig 2: cross-binary DF spectrum
fig,ax=plt.subplots(figsize=(3.4,2.2))
x=['1','2-3','4-22','23-43','44 (all)']; y=[5291,413,270,67,148]
b=ax.bar(x,y,color=['#1f77b4','#d62728','#ff7f0e','#ff7f0e','#9467bd'])
ax.set_yscale('log'); ax.set_ylabel('# function signatures'); ax.set_xlabel('# binaries sharing (document frequency)')
ax.set_title('Cross-binary DF spectrum (44 executables)',fontsize=8.5)
ax.annotate('unique (85%)',(0,5291),(0.2,2600),fontsize=7,ha='left')
ax.annotate('copy-paste\ncand.',(1,413),(1.0,1300),fontsize=7,ha='center')
ax.annotate('runtime',(4,148),(3.4,420),fontsize=7,ha='center')
plt.savefig('fig_df.pdf'); plt.close()

# Fig 3: prevalence per binary
fig,ax=plt.subplots(figsize=(3.4,2.2))
import numpy as np
bins=['amc','atf_unit','aqlite']; ex=[32.9,29.1,5.4]; nd=[38.9,33.2,6.9]
xp=np.arange(len(bins)); w=0.38
ax.bar(xp-w/2,ex,w,label='opcode-identical',color='#1f77b4')
ax.bar(xp+w/2,nd,w,label='near-dup families',color='#ff7f0e')
ax.set_xticks(xp); ax.set_xticklabels(bins); ax.set_ylabel('% of functions'); ax.legend(fontsize=7)
ax.set_title('Near-duplicate prevalence',fontsize=8.5)
plt.savefig('fig_prev.pdf'); plt.close()

# Fig 4: IDF discrimination
fig,ax=plt.subplots(figsize=(3.4,2.2))
labs=['app1~app2\n(copy-paste GT)','md5sum~sha256sum\n(shared gnulib)','grep~gzip\n(unrelated)']
vals=[63,958,11]; cols=['#2ca02c','#1f77b4','#bbbbbb']
b=ax.bar(labs,vals,color=cols)
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+15,str(v),ha='center',fontsize=8)
ax.set_ylabel('exclusive shared 12-grams'); ax.set_title('DF discriminator: signal vs noise floor',fontsize=8.5)
ax.tick_params(axis='x',labelsize=6.5)
plt.savefig('fig_idf.pdf'); plt.close()
print("wrote fig_compare.pdf fig_df.pdf fig_prev.pdf fig_idf.pdf")
