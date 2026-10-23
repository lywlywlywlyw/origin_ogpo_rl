"""Audit recorded executed latent RMS by episode position; no replay reconstruction."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('exp/ogpo/OGPO/square/square-mh-low_dim/anchorq_square_shared_ogpo_bc_v2seed1')
out=Path('run_logs/anchorq_episode_rms');out.mkdir(parents=True,exist_ok=True)
d=np.genfromtxt(root/'env.csv',delimiter=',',names=True)
steps=d['total_env_steps'].astype(int)
assert np.all(np.diff(np.r_[0,steps])==4)
assert np.max(d['successful_replay_chunks'])==0
# Wrapper resets t=0 and truncates at400; base robomimic is_done() is always False.
# Cross-check reset counts against logged anchor-bank constructions.
budget=np.genfromtxt(root/'online_budget.csv',delimiter=',',names=True)
assert np.all(budget['anchor_bank_constructions']==1+budget['decisions']+np.floor(budget['env_steps']/400))
t=(steps-4)%400; episode=(steps-4)//400
z=d['latent_rms'];card=d['anchor_index']<6;defer=~card;pre=d['particle_active']==0

def stats(mask):
 v=z[mask]
 return dict(n=int(mask.sum()),median=float(np.median(v)) if len(v) else None,
  p99=float(np.percentile(v,99)) if len(v) else None,max=float(v.max()) if len(v) else None,
  **{f'gt{n}':int((v>n).sum()) for n in [10,100,1000,10000]})
summary={'rows':len(d),'physical_steps':int(steps[-1]),'complete_episodes':int(steps[-1]//400),
 'partial_episode_steps':int(steps[-1]%400),'metric':'selected executed latent RMS, not all raw anchors',
 'reset_cardinal':stats(card&(t==0)),'reset_defer':stats(defer&(t==0)),
 'all_cardinal':stats(card),'all_defer':stats(defer),'pre_particle_cardinal':stats(card&pre),
 'post_particle_cardinal':stats(card&~pre),'bins':[], 'first_exceedances':{}}
for lo,hi in [(0,40),(40,100),(100,200),(200,300),(300,400)]:
 summary['bins'].append(dict(start=lo,end=hi,**stats(card&(t>=lo)&(t<hi))))
for threshold in [10,100,1000,10000]:
 records=[]
 for ep in np.unique(episode):
  found=np.flatnonzero((episode==ep)&card&(z>threshold))
  if len(found):
   i=found[0];records.append(dict(episode=int(ep),t=int(t[i]),rms=float(z[i]),global_step=int(steps[i]-4)))
 summary['first_exceedances'][str(threshold)]=records
np.savetxt(out/'rows.csv',np.column_stack([steps,episode,t,d['anchor_index'],d['particle_active'],z]),delimiter=',',
 header='global_end_step,episode_zero_based,episode_action_start_step,anchor_index,particle_active,executed_latent_rms',comments='')
(out/'summary.json').write_text(json.dumps(summary,indent=2))
fig,axs=plt.subplots(3,1,figsize=(11,12),constrained_layout=True)
for mask,label,color in [(card,'Selected direction (0-5)','#c25a32'),(defer,'Selected defer (6)','#2477aa')]:
 axs[0].scatter(t[mask],z[mask],s=3,alpha=.12,color=color,label=label,rasterized=True)
axs[0].set_yscale('log');axs[0].set_ylabel('Executed latent RMS (log scale)');axs[0].legend()
axs[0].set_title('AnchorQ cwo01uyp: logged RMS vs. episode action-start step')
for mask,label in [(card&pre,'Before particle activation'),(card&~pre,'After particle activation')]:
 bins=np.arange(0,401,20); rates=[]
 for lo,hi in zip(bins[:-1],bins[1:]):
  m=mask&(t>=lo)&(t<hi);rates.append(100*np.mean(z[m]>100) if m.any() else np.nan)
 axs[1].plot(bins[:-1]+10,rates,marker='o',label=label)
axs[1].set_ylabel('Direction selections with RMS >100 (%)');axs[1].legend()
worst=sorted(np.unique(episode),key=lambda ep:z[episode==ep].max(),reverse=True)[:4]
for ep in worst:
 m=(episode==ep)&card
 axs[2].plot(t[m],z[m],'.-',lw=.8,label=f'Episode {ep+1}')
axs[2].set_yscale('log');axs[2].set_ylabel('RMS, direction selections only');axs[2].legend()
for ax in axs:
 ax.set_xlim(-4,400);ax.set_xlabel('Physical steps since reset (latent chosen before executing next chunk)');ax.grid(alpha=.2)
fig.savefig(out/'episode_rms.png',dpi=180);fig.savefig(out/'episode_rms.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='first_exceedances'},indent=2))
for threshold,records in summary['first_exceedances'].items():
 a=np.array([r['t'] for r in records]);print('FIRST',threshold,'episodes',len(a),'min/median/max',np.min(a),np.median(a),np.max(a))
print('worst',worst)
