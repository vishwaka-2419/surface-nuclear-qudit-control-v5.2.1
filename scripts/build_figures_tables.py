"""Regenerate every figure, table and numerical macro from the archived results.

Nothing in this file computes physics: it only reads results/*.json|csv, draws,
and writes figures/, figure_data/, supplement/generated/ and
manuscript/generated/numbers.tex.  Run after any numerical study.
"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,NullFormatter,FixedLocator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from qudit_control.io import load_pulse
from qudit_control.validation import envelope_function
from qudit_control.model import MU_B
from qudit_control.requirements import electron_splitting,K_PER_MHZ
R=ROOT/'results';F=ROOT/'figures';D=ROOT/'figure_data'
for p in (F,D,ROOT/'manuscript/generated',ROOT/'supplement/generated'):p.mkdir(exist_ok=True,parents=True)

# ---------------------------------------------------------------- style
C=dict(blue='#1F5FA8',orange='#E4572E',teal='#139C8A',purple='#8E4DB8',gold='#B8860B',
       gray='#6C757D',light='#C9D1DA',ink='#22303F',grid='#E6EAEF')
SEQ=['#A9CBE8','#7FB0DB','#5A95CC','#3E7CBF','#2763A8','#174E8C','#0E3A6B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,
    'legend.fontsize':6.6,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'mathtext.fontset':'dejavusans',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.major.width':.6,
    'ytick.major.width':.6,'xtick.minor.width':.4,'ytick.minor.width':.4,'xtick.major.size':2.8,
    'ytick.major.size':2.8,'xtick.minor.size':1.6,'ytick.minor.size':1.6,'lines.linewidth':1.4,
    'lines.markersize':4,'savefig.dpi':400,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'axes.labelcolor':C['ink'],'text.color':C['ink'],'axes.edgecolor':C['gray'],'xtick.color':C['ink'],
    'ytick.color':C['ink'],'legend.frameon':False,'legend.handlelength':1.6,'legend.handletextpad':.5,
    'legend.borderaxespad':.2,'legend.labelspacing':.35,'axes.titleweight':'medium'})
COL=3.375   # REVTeX column width (in)
manifest=[]
def j(name):return json.loads((R/(name+'.json')).read_text())
def grid(a,axis='y'):a.set_axisbelow(True);a.grid(axis=axis,color=C['grid'],lw=.5)
def label(a,s,x=-.22,y=1.04):a.text(x,y,s,transform=a.transAxes,fontsize=8.5,fontweight='bold',va='bottom',ha='left')
def save(f,name,data=None,sources=()):
    for ext in ('pdf','png','svg'):f.savefig(F/f'{name}.{ext}',bbox_inches='tight',pad_inches=.02)
    if isinstance(data,pd.DataFrame):data.to_csv(D/f'{name}.csv',index=False)
    plt.close(f);manifest.append(dict(figure=name,source_files=';'.join(sources),data_csv=f'figure_data/{name}.csv' if data is not None else ''))
def tex(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def table(name,caption,cols,rows,align=None,label=None,size=r'\small',colsep='4pt',resize=False):
    width=align or 'l'+'r'*(len(cols)-1)
    out=[r'\begin{table}[htbp]',r'\centering',size,r'\setlength{\tabcolsep}{'+colsep+'}',r'\caption{'+caption+'}'+(r'\label{'+label+'}' if label else '')]
    if resize:out.append(r'\resizebox{\textwidth}{!}{%')
    out+=[r'\begin{tabular}{'+width+'}',r'\toprule',' & '.join(cols)+r' \\',r'\midrule']
    out+=[' & '.join(map(str,row))+r' \\' for row in rows]
    out+=[r'\bottomrule',r'\end{tabular}']
    if resize:out.append('}')
    out.append(r'\end{table}')
    (ROOT/'supplement/generated'/f'{name}.tex').write_text('\n'.join(out)+'\n')
def pct(x,n=3):return f'{100*x:.{n}f}\\%'
def pctfloor(x,n=3):
    # rounded down, for 'at least' statements
    import math;return f'{math.floor(100*x*10**n)/10**n:.{n}f}\\%'
nomp=j('nominal_N_B1.6_dual')['config']['parameters']
from qudit_control.model import Parameters
P=Parameters(**nomp)
GAMMA=P.gamma;NU16=electron_splitting(1.6,P);OMEGA_PER_U=MU_B*P.g_t/GAMMA   # electron Rabi per unit u

# ================================================================ FIGURE 1: overview
comb=pd.read_csv(R/'comb_spectrum.csv')
f=plt.figure(figsize=(COL,4.35),layout='constrained')
gs=f.add_gridspec(2,2,height_ratios=[1.05,1.25],width_ratios=[1,1.05])
# (a) schematic -----------------------------------------------------------------
a=f.add_subplot(gs[0,0]);a.set_axis_off();a.set_xlim(0,10);a.set_ylim(0,10)
a.add_patch(patches.Rectangle((0,0),10,2.2,color='#B8C0C8',lw=0))              # Ag(100)
a.add_patch(patches.Rectangle((0,2.2),10,1.6,color='#E4ECF4',lw=0))            # MgO
for x in np.arange(.6,10,1.25):
    a.add_patch(patches.Circle((x,3.0),.28,color='#9CB4CC',lw=0))
a.text(5,1.0,'Ag(100)',ha='center',va='center',fontsize=6.5,color=C['ink'])
a.text(5,2.95,'MgO',ha='center',va='center',fontsize=6.5,color=C['ink'])
a.add_patch(patches.Circle((5,4.6),.8,color=C['blue'],alpha=.9,lw=0))              # Ti electron cloud
a.add_patch(patches.Circle((5,4.6),.27,color=C['orange'],lw=0))                     # nucleus
a.annotate('',xy=(6.55,5.55),xytext=(6.55,4.25),arrowprops=dict(arrowstyle='-|>',color=C['blue'],lw=1.1,mutation_scale=7))
a.text(6.85,4.95,'$S=1/2$',fontsize=6.6,color=C['blue'],va='center')
a.annotate('',xy=(3.45,5.1),xytext=(3.45,4.2),arrowprops=dict(arrowstyle='-|>',color=C['orange'],lw=1.0,mutation_scale=6))
a.text(3.2,4.65,'$I=7/2$',fontsize=6.6,color=C['orange'],va='center',ha='right')
tip=patches.Polygon([[3.7,10],[6.3,10],[5.0,6.3]],closed=True,color='#7D8A97',lw=0);a.add_patch(tip)
a.text(5.0,8.5,'tip',ha='center',va='center',fontsize=6.5,color='white')
t=np.linspace(0,1,80);a.plot(0.5+2.1*t,8.0+.45*np.sin(2*np.pi*4*t),color=C['teal'],lw=1.1)
a.text(1.55,9.0,'$V_{\\rm RF}$',fontsize=7,color=C['teal'],ha='center')
a.annotate('',xy=(9.0,9.4),xytext=(9.0,6.6),arrowprops=dict(arrowstyle='-|>',color=C['ink'],lw=.9,mutation_scale=7))
a.text(9.4,8.0,'$B$',fontsize=7,va='center')
label(a,'(a)',x=-.06,y=.97)
# (b) level scheme ---------------------------------------------------------------
b=f.add_subplot(gs[0,1]);b.set_axis_off();b.set_xlim(-1.2,11);b.set_ylim(-.8,10.6)
E=np.array([j('nominal_N_B1.6_dual')['config']['B']]);pr,x,_=load_pulse(ROOT,'nominal_nuclear_B1.6_single')
Eg=pr.E[0];Ee=pr.E[1]
def ladder(x0,Es,col,lab):
    y=0.9+7.2*(Es-Es.min())/(Es.max()-Es.min())
    for yy in y:b.plot([x0,x0+2.4],[yy,yy],color=col,lw=.9)
    for k in range(7):
        b.annotate('',xy=(x0+1.2,y[k+1]),xytext=(x0+1.2,y[k]),arrowprops=dict(arrowstyle='<->',color=C['teal'],lw=.6,mutation_scale=4,shrinkA=0,shrinkB=0))
    b.text(x0+1.2,9.05,lab,ha='center',fontsize=6.8,color=col)
    return y
yg=ladder(1.2,Eg,C['blue'],'$s=g$');ye=ladder(6.4,Ee,C['orange'],'$s=e$')
b.text(1.05,yg[0],'$-7/2$',fontsize=5.8,ha='right',va='center',color=C['gray'])
b.text(1.05,yg[-1],'$m=+7/2$',fontsize=5.8,ha='right',va='center',color=C['gray'])
b.text(8.95,ye[0],'$+7/2$',fontsize=5.8,ha='left',va='center',color=C['gray'])
b.text(8.95,ye[-1],'$-7/2$',fontsize=5.8,ha='left',va='center',color=C['gray'])
b.annotate('',xy=(5.0,8.0),xytext=(5.0,2.0),arrowprops=dict(arrowstyle='<->',color=C['gray'],lw=.7,mutation_scale=6))
b.text(5.0,5.0,'$\\nu_e$\n13 GHz',fontsize=6,color=C['gray'],ha='center',va='center',bbox=dict(boxstyle='square,pad=0.15',fc='white',ec='none'))
b.text(2.4,-.3,'7 carriers',ha='center',fontsize=6.2,color=C['teal']);b.text(7.6,-.3,'7 carriers',ha='center',fontsize=6.2,color=C['teal'])
b.text(5.0,10.1,'order reversed',ha='center',fontsize=5.8,color=C['gray'])
label(b,'(b)',x=-.06,y=.97)
# (c) comb spectrum --------------------------------------------------------------
c=f.add_subplot(gs[1,:]);grid(c)
for br,col in (('ground',C['blue']),('excited',C['orange'])):
    g=comb[comb.branch==br]
    for k,(lk,gg) in enumerate(g.groupby('link')):
        c.plot(gg.B_T,gg.f_MHz,color=col,lw=.9,label=('lower state ($s=g$)' if br=='ground' else 'upper state ($s=e$)') if k==0 else None)
for Bv,txt,yy in ((1.03,'1.03 T',80.8),(1.35,'1.35 T (cw readout)',80.8),(1.6,'1.6 T (pulsed readout)',82.9),(1.714,'1.714 T',80.8)):
    c.axvline(Bv,color=C['gray'],ls=':',lw=.7);c.text(Bv,yy,txt,ha='center',va='bottom',fontsize=5.6,color=C['ink'])
c.set_xlim(.6,1.9);c.set_ylim(48,80.5);c.set_xlabel('External field $B$ (T)');c.set_ylabel('Nuclear transition\nfrequency (MHz)')
c.legend(loc='lower left',ncol=2,fontsize=6.3,handlelength=1.3,bbox_to_anchor=(-.01,-.03))
label(c,'(c)',x=-.17,y=1.0)
save(f,'fig1_overview',comb,['comb_spectrum.csv','nominal_nuclear_B1.6_single.json'])

# ================================================================ FIGURE 2: optimized gates
names=[f'nominal_N_B{b}_dual' for b in ('1.03','1.35','1.6','1.714')]+['nominal_nuclear_B1.6_single']
labels=['1.03 T\n14 tones','1.35 T\n14 tones','1.6 T\n14 tones','1.714 T\n14 tones','1.6 T\n7 tones']
rows=[]
for n in names:
    v=j('validation_'+n);rows.append(dict(name=n,optimized_full_fidelity=v['full_electron'][-1]['worst'],plain_projected_fidelity=v['plain_multitone']['worst'],
        peak_tone_kHz=1e3*v['resources']['peak_tone_MHz'],halfband_MHz=v['resources']['envelope_99pct_power_halfband_MHz']))
nom=pd.DataFrame(rows)
f=plt.figure(figsize=(COL,4.1),layout='constrained');gs=f.add_gridspec(2,1,height_ratios=[1,1.05])
a=f.add_subplot(gs[0]);grid(a)
pr,x,_=load_pulse(ROOT,'nominal_nuclear_B1.6_single');env=envelope_function(pr,x);tt=np.linspace(0,pr.T,801);aa=np.abs(env(tt))*1e3
order=np.argsort(pr.freq)
for rank,k in enumerate(order):
    a.plot(tt,aa[:,k],color=SEQ[rank],lw=1.1,label=f'{pr.freq[k]:.2f} MHz')
a.set_xlabel('Time ($\\mu$s)');a.set_ylabel('Envelope $|a_j(t)|$ (kHz)');a.set_xlim(0,pr.T);a.set_ylim(0,52)
a.axhline(50,color=C['gray'],ls='--',lw=.7);a.text(pr.T-.1,50.6,'bound $a_{\\max}$',ha='right',va='bottom',fontsize=6,color=C['gray'])
a.legend(title='carrier frequency',title_fontsize=6.2,fontsize=5.9,ncol=4,loc='upper center',bbox_to_anchor=(.5,.93),handlelength=1.1,columnspacing=.9)
label(a,'(a)',x=-.2,y=1.0)
envdata=pd.DataFrame({'time_us':tt});envdata=pd.concat([envdata,pd.DataFrame(aa,columns=[f'tone_{k}_{pr.freq[k]:.3f}MHz_kHz' for k in range(pr.J)])],axis=1)
b=f.add_subplot(gs[1]);grid(b);xx=np.arange(len(nom))
for i,row in nom.iterrows():b.plot([i,i],[1-row.plain_projected_fidelity,1-row.optimized_full_fidelity],color=C['light'],lw=1,zorder=1)
b.semilogy(xx,1-nom.plain_projected_fidelity,'s',color=C['orange'],ms=5,label='unshaped multitone pulse',zorder=3)
b.semilogy(xx,1-nom.optimized_full_fidelity,'o',color=C['blue'],ms=5,label='optimized pulse (full 16-level model)',zorder=3)
b.axhline(1e-2,color=C['gray'],ls='--',lw=.7);b.text(4.45,1.25e-2,'1% error',ha='right',va='bottom',fontsize=6,color=C['gray'])
b.set_xticks(xx,labels);b.set_ylabel('Gate infidelity $1-F$');b.set_ylim(3e-6,1);b.set_xlim(-.5,4.5)
b.yaxis.set_major_locator(LogLocator(base=10,numticks=8));b.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*.1,numticks=20));b.yaxis.set_minor_formatter(NullFormatter())
b.legend(loc='lower left',fontsize=6.3)
label(b,'(b)',x=-.2,y=1.0)
save(f,'fig2_gates',nom,['validation_'+n+'.json' for n in names]);envdata.to_csv(D/'fig2_gates_envelopes.csv',index=False)

# ================================================================ FIGURE 3: driving through the electron
gains=pd.read_csv(R/'operator_gains.csv')
def shift_record(name):
    p=R/f'validation_{name}.json'
    if not p.exists():return None
    v=json.loads(p.read_text());u2=v['resources']['real_waveform_rms_MHz']**2
    so=v.get('second_order_physical')
    sh=[abs(np.diff(s)).mean()*1e3 for s in so['mean_diagonal_shift_MHz']] if so else None
    return dict(name=name,u2_MHz2=u2,omega2_MHz2=u2*OMEGA_PER_U**2,shift_kHz=max(sh) if sh else None,
                linear_projected=v['physical_projected']['worst'] if 'physical_projected' in v else v.get('linear_projected',{}).get('worst'),
                full=v['full_electron'][-1]['worst'],second_order=so['worst'] if so else None)
cases=[('electron','single','$O_E$, 7 tones'),('electron','dual','$O_E$, 14 tones'),('magnetic','single','$O_M$, 7 tones'),('magnetic','dual','$O_M$, 14 tones')]
oprows=[]
for mech,act,lab in cases:
    lin=shift_record(f'nominal_{mech}_B1.6_{act}');trk=shift_record(f'tracked_{mech}_B1.6_{act}');old=shift_record(f'stark_{mech}_B1.6_{act}')
    oprows.append(dict(label=lab,mechanism=mech,tones=7 if act=='single' else 14,linear_projected=lin['linear_projected'],linear_full=lin['full'],
        linear_u2=lin['u2_MHz2'],linear_shift_kHz=(shift_record(f'stark_{mech}_B1.6_{act}') or {}).get('shift_kHz'),
        tracked_full=trk['full'] if trk else np.nan,tracked_u2=trk['u2_MHz2'] if trk else np.nan,tracked_shift_kHz=trk['shift_kHz'] if trk else np.nan,
        tracked_second_order=trk['second_order'] if trk else np.nan,untracked_second_order_design_full=old['full'] if old else np.nan))
op=pd.DataFrame(oprows)
# exact second-order slope at 1.6 T from K (recomputed here from the archived pulse problem; pure algebra)
from qudit_control.extended import ExtendedProblem
from qudit_control.virtual_electron import second_order_blocks
prE=ExtendedProblem(B=1.6,p=P,T=12.,amax=.05,ncoef=14,dt=.05,active=(0,1),mechanism='electron')
K=second_order_blocks(prE);slope_exact=np.mean([np.abs(np.diff(np.real(np.diag(K[s])))) for s in (0,1)])  # MHz per MHz^2 of u^2
slope_analytic=OMEGA_PER_U**2*P.A_z/(4*NU16**2)
f=plt.figure(figsize=(COL,5.3),layout='constrained');gs=f.add_gridspec(2,2,height_ratios=[1,1.15])
a=f.add_subplot(gs[0,0]);grid(a)
for mech,col,lab in (('electron',C['orange'],'$O_E$'),('magnetic',C['purple'],'$O_M$')):
    for s_,ls,sl in ((0,'-','lower'),(1,'--','upper')):
        g=gains[(gains.mechanism==mech)&(gains.branch==s_)]
        a.plot(g.B_T,g.gain_mean,color=col,ls=ls,lw=1.2)
        a.fill_between(g.B_T,g.gain_min,g.gain_max,color=col,alpha=.15,lw=0)
        a.text(g.B_T.iloc[-1]+.02,g.gain_mean.iloc[-1],f'{lab}, {sl}',color=col,fontsize=5.6,va='center',ha='left')
a.plot([.8,1.8],[1,1],color=C['blue'],lw=1,ls=':');a.text(1.82,1.0,'$O_N$ (both)',color=C['blue'],ha='left',va='center',fontsize=5.6)
a.axhline(0,color=C['gray'],lw=.5)
a.set_xlabel('External field $B$ (T)');a.set_ylabel('Projected coupling / $I_x$ coupling');a.set_xlim(.8,2.15);a.set_ylim(-8.5,9);a.set_xticks([.8,1.2,1.6])
label(a,'(a)',x=-.3,y=1.0)
b=f.add_subplot(gs[0,1]);grid(b)
om=np.linspace(0,420,200)
b.plot(om,1e3*slope_exact*om**2/OMEGA_PER_U**2,color=C['ink'],lw=1.3,label='exact second order, Eq. (3)')
b.plot(om,1e3*slope_analytic*om**2/OMEGA_PER_U**2,color=C['gold'],ls=(0,(3,2)),lw=1.1,label='leading order, Eq. (5)')
textpos={'$O_E$, 7 tones':(60,12),'$O_E$, 14 tones':(120,27),'$O_M$, 7 tones':(60,22),'$O_M$, 14 tones':(175,35)}
for row in op.itertuples():
    if not np.isnan(row.tracked_u2):
        xo=np.sqrt(row.tracked_u2)*OMEGA_PER_U;b.plot(xo,row.tracked_shift_kHz,'o' if row.mechanism=='electron' else 'D',color=C['blue'] if row.tones==7 else C['teal'],ms=4.5,zorder=4)
        b.annotate(row.label.replace(' tones',''),xy=(xo,row.tracked_shift_kHz),xytext=textpos[row.label],fontsize=5.6,ha='right',va='center',arrowprops=dict(arrowstyle='-',color=C['gray'],lw=.6,shrinkB=3))
b.set_xlabel('rms electron Rabi frequency\n$\\langle\\Omega_e^2\\rangle^{1/2}$ (MHz)');b.set_ylabel('Comb shift $\\Delta f$ (kHz)');b.set_xlim(0,420);b.set_ylim(0,42)
b.legend(fontsize=5.8,loc='upper left',handlelength=1.6)
label(b,'(b)',x=-.3,y=1.0)
c=f.add_subplot(gs[1,:]);grid(c);xx=np.arange(len(op))
c.semilogy(xx-.22,1-op.linear_projected,'s',color=C['teal'],ms=5,label='design without shift, evaluated in its own (projected) model')
c.semilogy(xx,1-op.linear_full,'o',color=C['orange'],ms=5,label='design without shift, full 16-level model')
c.semilogy(xx+.22,1-op.tracked_full,'D',color=C['blue'],ms=5,label='shift tracked in carriers and design, full 16-level model')
for i,row in op.iterrows():c.plot([i-.22,i],[1-row.linear_projected,1-row.linear_full],color=C['light'],lw=.9,zorder=1)
c.axhline(1e-2,color=C['gray'],ls='--',lw=.7);c.text(3.45,1.3e-2,'1% error',ha='right',va='bottom',fontsize=6,color=C['gray'])
c.set_xticks(xx,op.label);c.set_ylabel('Gate infidelity $1-F$');c.set_ylim(2e-6,1.5);c.set_xlim(-.6,3.6)
c.yaxis.set_major_locator(LogLocator(base=10,numticks=8));c.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*.1,numticks=20));c.yaxis.set_minor_formatter(NullFormatter())
c.legend(loc='upper left',bbox_to_anchor=(.0,.9),fontsize=6.0,handlelength=1.3,frameon=True,framealpha=.95,edgecolor='none')
label(c,'(c)',x=-.14,y=1.0)
save(f,'fig3_electron',op,['operator_gains.csv','validation_nominal_*_B1.6_*.json','validation_tracked_*_B1.6_*.json','validation_stark_*_B1.6_*.json'])

# ================================================================ FIGURE 4: calibration tolerance
rob_names=['nominal_N_B1.6_dual','nominal_nuclear_B1.6_single','robust_N_B1.6_single_seed1','robust_N_B1.6_dual_corners','robust_N_B1.6_dual_refined']
rob_labels=['12 $\\mu$s\n14 tones\nnominal','12 $\\mu$s\n7 tones\nnominal','48 $\\mu$s\n7 tones\naxis-trained','48 $\\mu$s\n14 tones\ncorner-trained','48 $\\mu$s\n14 tones\nrefined']
rv=[]
for n,lab in zip(rob_names,rob_labels):
    p=R/('robust_validation_'+n+'.json')
    if not p.exists():continue
    v=json.loads(p.read_text());rv.append(dict(name=n,label=lab,nominal=v['nominal_physical']['worst'],corner=v['selected_corner_physical']['worst'],heldout=v['heldout_rwa_min'],corner_rwa=v['corner_rwa_min']))
rob=pd.DataFrame(rv)
selected=[('nominal_N_B1.6_dual','nominal, 12 $\\mu$s, 14 tones',C['gray'],'-'),('robust_N_B1.6_single_seed1','robust, 48 $\\mu$s, 7 tones',C['teal'],'-'),('robust_N_B1.6_dual_refined','robust, 48 $\\mu$s, 14 tones',C['blue'],'-')]
f=plt.figure(figsize=(COL,4.9),layout='constrained');gs=f.add_gridspec(2,2,height_ratios=[1,1.1])
scans=[]
for k,(axis,scale,xlabel) in enumerate([('gain',100,'Drive-gain error (%)'),('A_z',1000,'Hyperfine offset $\\delta A_z$ (kHz)')]):
    a=f.add_subplot(gs[0,k]);grid(a)
    for n,lab,col,ls in selected:
        p=R/f'axis_test_{n}.csv'
        if not p.exists():continue
        g=pd.read_csv(p);g=g[g.axis==axis].copy();g['name']=n;scans.append(g)
        a.semilogy(g.offset*scale,np.maximum(1-g.fidelity,1e-9),color=col,ls=ls,lw=1.3,label=lab)
    a.axhline(1e-2,color=C['orange'],ls='--',lw=.8)
    a.set_xlabel(xlabel);a.set_ylim(5e-8,.5);a.yaxis.set_minor_formatter(NullFormatter())
    if k==0:a.set_ylabel('Gate infidelity $1-F$')
    else:a.tick_params(labelleft=False)
    if k==0:a.text(-5.9,6.5e-3,'1% error',fontsize=6,color=C['orange'],va='top',ha='left')
    label(a,'(a)' if k==0 else '(b)',x=-.32 if k==0 else -.12,y=1.0)
    if k==1:a.legend(fontsize=6,loc='upper center',bbox_to_anchor=(-.1,1.36),ncol=1,handlelength=1.6)
c=f.add_subplot(gs[1,:]);grid(c);xx=np.arange(len(rob))
for dx,key,col,mkr,lab in [(-.2,'nominal',C['teal'],'o','nominal (real waveform)'),(0.,'heldout',C['blue'],'s','worst of 128 held-out interior points'),(.2,'corner',C['orange'],'D','worst corner of Eq. (6) (real waveform)')]:
    c.semilogy(xx+dx,1-rob[key],mkr,color=col,ms=5,label=lab)
c.axhline(1e-2,color=C['gray'],ls='--',lw=.7);c.text(4.45,1.3e-2,'1% error',ha='right',va='bottom',fontsize=6,color=C['gray'])
c.set_xticks(xx,rob.label,fontsize=6);c.set_ylabel('Gate infidelity $1-F$');c.set_ylim(3e-6,.2);c.set_xlim(-.5,4.5)
c.yaxis.set_minor_formatter(NullFormatter());c.legend(loc='lower right',fontsize=6,handlelength=1.2)
label(c,'(c)',x=-.14,y=1.0)
save(f,'fig4_calibration',rob,['robust_validation_*.json','axis_test_*.csv']);pd.concat(scans,ignore_index=True).to_csv(D/'fig4_calibration_scans.csv',index=False)

# ================================================================ FIGURE 5: electron state
noise=pd.read_csv(R/'prepared_telegraph.csv');req=pd.read_csv(R/'polarization_requirement_map.csv');rr=pd.read_csv(R/'operating_requirements.csv')
pt=float(rr.query('B_T==1.6 and T_us==12').p_thermal_04K.iloc[0]);pe_req=float(rr.query('B_T==1.6 and T_us==12').p_exc_sufficient.iloc[0])
f=plt.figure(figsize=(COL,4.6),layout='constrained');gs=f.add_gridspec(2,1,height_ratios=[1,1.05])
a=f.add_subplot(gs[0]);grid(a)
for k,(pe,g) in enumerate(noise.groupby('p_exc',sort=False)):
    col=[C['orange'],C['blue'],C['teal']][k%3];mkr=['o','s','D'][k%3]
    lab=r'$p_e=0.17$ (thermal, 0.4 K)' if pe>.1 else r'$p_e=10^{'+str(int(round(np.log10(pe))))+'}$'
    a.errorbar(g.tau_c_us,1-g.fidelity,yerr=g['sem'],marker=mkr,color=col,capsize=1.5,label=lab,lw=1.2,ms=4)
a.set_xscale('log');a.set_yscale('log');a.axvline(.1,color=C['gray'],ls=':',lw=.8);a.text(.115,2.5e-6,'reported electron\nlifetime (readout)',fontsize=5.8,color=C['gray'],va='bottom')
a.axhline(1e-2,color=C['gray'],ls='--',lw=.7);a.text(8e3,1.3e-2,'1% error',ha='right',va='bottom',fontsize=6,color=C['gray'])
a.set_xlabel(r'Electron correlation time $\tau_c$ ($\mu$s)');a.set_ylabel('Gate infidelity $1-F$');a.set_ylim(1e-6,2);a.set_xlim(.06,1.5e4)
a.legend(fontsize=6,loc='upper center',bbox_to_anchor=(.5,1.17),ncol=3,handlelength=1.4,columnspacing=1.0);a.yaxis.set_minor_formatter(NullFormatter())
label(a,'(a)',x=-.2,y=1.08)
b=f.add_subplot(gs[1]);grid(b)
for T,col in zip([12.,48.,250.],[C['blue'],C['teal'],C['purple']]):
    g=req[np.isclose(req.T_us,T)&(req.p_exc_sufficient<=.5)];b.loglog(g.tau_c_us,g.p_exc_sufficient,color=col,lw=1.4,label=f'$T={T:g}\\,\\mu$s')
b.axhline(pt,color=C['orange'],ls='--',lw=1,label='thermal electron at 0.4 K')
b.axvline(.1,color=C['gray'],ls=':',lw=.8);b.plot([.1],[pe_req],'o',color=C['blue'],ms=5,zorder=5)
b.annotate(f'${pe_req*1e5:.1f}\\times10^{{-5}}$',xy=(.1,pe_req),xytext=(1.6e-3,1.2e-3),fontsize=6.3,arrowprops=dict(arrowstyle='-',color=C['gray'],lw=.7))
b.set_xlabel(r'Electron correlation time $\tau_c$ ($\mu$s)');b.set_ylabel('Sufficient bound on $p_e$');b.set_ylim(1e-8,.8);b.set_xlim(1e-4,1e4)
b.legend(fontsize=6,loc='lower right',handlelength=1.4);b.yaxis.set_minor_formatter(NullFormatter())
# right axis: thermal-equivalent temperature at 1.6 T (same quantity, converted)
def pe_to_T(pe):return NU16*K_PER_MHZ/np.log((1-pe)/pe)*1e3
def T_to_pe(T):
    with np.errstate(divide='ignore',over='ignore'):
        return 1/(1+np.exp(NU16*K_PER_MHZ/(np.asarray(T)/1e3)))
sec=b.secondary_yaxis('right',functions=(lambda p:pe_to_T(np.clip(p,1e-12,.499999)),T_to_pe))
sec.set_ylabel('Thermal-equivalent $T_e$ at 1.6 T (mK)',fontsize=7)
sec.yaxis.set_major_locator(FixedLocator([40,50,60,70,80,100,150,200,300]));sec.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f'{v:g}'));sec.yaxis.set_minor_locator(FixedLocator([]));sec.tick_params(labelsize=6.3)
sec.spines['right'].set_visible(True)
label(b,'(b)',x=-.2,y=1.0)
save(f,'fig5_electron_state',noise,['prepared_telegraph.csv','polarization_requirement_map.csv','operating_requirements.csv']);req.to_csv(D/'fig5_requirement_map.csv',index=False)

# ================================================================ SUPPLEMENT FIGURES
def sfig(h=2.5,w=4.6):
    f,a=plt.subplots(figsize=(w,h),layout='constrained');grid(a);return f,a
unc=pd.read_csv(R/'source_uncertainty_intervals.csv');f,a=sfig(h=2.3,w=4.4);labels_unc=[]
for i,row in unc.iterrows():
    labels_unc.append((r'$\sigma_{\kappa47}=0.8$ MHz' if row.sigma_k47_MHz<1 else r'$\sigma_{\kappa47}=16$ MHz')+('\nbulk $g_N$' if row.source_gamma=='bulk_coefficient' else '\nsurface-fit $g_N$'))
    a.errorbar(row.B_r1_T,i,xerr=[[row.B_r1_T-row.B_lower_T],[row.B_upper_T-row.B_r1_T]],fmt='o',capsize=3,color=C['blue'] if row.source_gamma=='bulk_coefficient' else C['orange'],ms=4)
a.set_yticks(range(4),labels_unc,fontsize=6.5);a.invert_yaxis();a.set_xlabel(r'Field of the $r=1$ comb-separation maximum (T)');a.set_xlim(.5,1.4);a.grid(axis='y',visible=False);grid(a,'x')
save(f,'source_uncertainty',unc,['source_uncertainty_intervals.csv'])

spec=pd.read_csv(R/'field_spectrum_new.csv');f,a=sfig(h=2.6)
labels_s={'anisotropic':'anisotropic nominal ($g_t=1.67$)','isotropic':r'isotropic ($g_t=g_z=0.56$)','surface_gamma':r'surface-fit $g_N=0.37$'}
for k,(sc,g) in enumerate(spec.groupby('scenario',sort=False)):a.plot(g.B_T,g.inter_MHz,color=[C['blue'],C['teal'],C['orange']][k],ls=['-','--',':'][k],label=labels_s.get(sc,sc))
for b_ in (1.03,1.35,1.6,1.714):a.axvline(b_,color=C['gray'],ls=':',lw=.6)
a.set_xlabel('External field $B$ (T)');a.set_ylabel('Nearest inter-comb\nseparation (MHz)');a.set_ylim(0,1.9);a.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=6.3)
save(f,'spectral_sensitivity',spec,['field_spectrum_new.csv'])

efg=pd.read_csv(R/'efg_spectral_stress.csv');f,a=sfig(h=2.6)
for (eta,tilt),col,ls in zip([(0,0),(.5,0),(0,15),(.2,15)],[C['blue'],C['teal'],C['orange'],C['purple']],['-','--',':','-.']):
    g=efg[np.isclose(efg.eta,eta)&np.isclose(efg.tilt_deg,tilt)];a.plot(g.B_T,g.inter_MHz,color=col,ls=ls,label=rf'$\eta={eta:g}$, tilt ${tilt:g}^\circ$')
a.set_xlabel('External field $B$ (T)');a.set_ylabel('Nearest inter-comb\nseparation (MHz)');a.set_ylim(0,1.9);a.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=6.3)
save(f,'efg_sensitivity',efg,['efg_spectral_stress.csv'])

frames=pd.read_csv(R/'phase_tracking_populations.csv');f,a=sfig(h=2.3)
a.plot(frames.idle_us*1e3,frames.untracked_population_distance,color=C['orange'],lw=1,label='carrier phases restarted after the idle')
a.plot(frames.idle_us*1e3,frames.tracked_population_distance,color=C['blue'],lw=1.8,label='carrier phases tracked continuously')
a.set_xlabel('Inserted idle time (ns)');a.set_ylabel('Population $L_1$ distance\nbetween electron branches');a.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=6.3)
save(f,'frame_tracking',frames,['phase_tracking_populations.csv'])

bb=pd.read_csv(R/'analytic_composite_baseline.csv');pos=bb[bb.gain_error>0];f,a=sfig(h=2.5,w=4.2)
for key,col,lab in [('plain_infidelity',C['orange'],'single rotation'),('BB1_infidelity',C['blue'],'ideal resonant BB1 (9$\\times$ area)')]:
    g=pos[pos[key]>2e-14];a.loglog(100*g.gain_error,g[key],color=col,label=lab)
a.axvline(1,color=C['gray'],ls=':',lw=.8);a.set_xlabel('Systematic amplitude error (%)');a.set_ylabel('Infidelity $1-F$');a.legend(loc='upper left',fontsize=6.3);a.set_ylim(1e-14,1);a.yaxis.set_minor_formatter(NullFormatter())
save(f,'composite_baseline',bb,['analytic_composite_baseline.csv'])

con=[]
for n,lab in [('nominal_N_B1.6_dual','$O_N$, 14 tones'),('nominal_electron_B1.6_single','$O_E$, 7 tones'),('nominal_magnetic_B1.6_single','$O_M$, 7 tones'),('tracked_electron_B1.6_single','$O_E$, 7 tones, tracked'),('tracked_magnetic_B1.6_single','$O_M$, 7 tones, tracked')]:
    p=R/f'validation_{n}.json'
    if not p.exists():continue
    for r_ in json.loads(p.read_text())['full_electron']:con.append(dict(name=n,label=lab,dt_ps=r_['dt_us']*1e6,fidelity=r_['worst']))
cv=pd.DataFrame(con);f,a=sfig(h=2.6,w=4.4)
for (n,g),col,mk_ in zip(cv.groupby('name',sort=False),[C['blue'],C['orange'],C['purple'],C['teal'],C['gold']],['o','s','D','^','v']):
    g=g.sort_values('dt_ps');ref=g.fidelity.iloc[0];gg=g.iloc[1:];a.loglog(gg.dt_ps,np.maximum(abs(gg.fidelity-ref),1e-14),marker=mk_,color=col,label=g.label.iloc[0],ms=4)
a.set_xlabel('Full-system midpoint step (ps)');a.set_ylabel('Fidelity change relative\nto the finest step');a.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=6.3)
a.xaxis.set_major_locator(FixedLocator([12.5,25,50,100,200]));a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f'{v:g}'));a.xaxis.set_minor_formatter(NullFormatter())
save(f,'full_system_convergence',cv,['validation_nominal_*_B1.6*.json','validation_tracked_*_B1.6*.json'])

hist=[];f,a=sfig(h=2.5)
for k,s in enumerate((1,2,3)):
    n=f'robust_N_B1.6_dual_seed{s}';v=j(n);h=pd.DataFrame(v['history']);h['name']=n;hist.append(h)
    a.semilogy(h.evaluation,1-h.worst,color=[C['blue'],C['teal'],C['purple']][k],label=f'independent start {s}',lw=1.2)
a.set_xlabel('Objective and gradient evaluations');a.set_ylabel('Worst training infidelity\n(axis ensemble, 48 $\\mu$s)');a.legend(loc='upper right',fontsize=6.3);a.yaxis.set_minor_formatter(NullFormatter())
save(f,'optimization_histories',pd.concat(hist,ignore_index=True),[f'robust_N_B1.6_dual_seed{s}.json' for s in (1,2,3)])

noisephi=pd.read_csv(R/'prepared_dephasing.csv');f,a=sfig(h=2.5,w=4.3)
for kind,col,lab,mk_ in [('magnetic',C['blue'],r'$L\propto I_z$ (adjacent coherence time)','o'),('quadrupolar',C['purple'],r'$L\propto I_z^2$ (outermost adjacent coherence time)','D')]:
    g=noisephi[noisephi.kind==kind];a.loglog(g.Tnoise_us/1000,1-g.fidelity,marker=mk_,color=col,label=lab,ms=4)
a.axhline(.01,color=C['gray'],ls='--',lw=.8);a.text(90,.0125,'1% error',ha='right',fontsize=6,color=C['gray'])
a.set_xlabel('Dephasing time (ms)');a.set_ylabel('Gate infidelity $1-F$');a.legend(loc='upper right',fontsize=6.3);a.yaxis.set_minor_formatter(NullFormatter())
save(f,'dephasing_comparison',noisephi,['prepared_dephasing.csv'])

dims=pd.read_csv(R/'dimension_transfer.csv');f,a=sfig(h=2.2,w=3.6)
a.semilogy(dims.d,1-dims.physical_fidelity,'o-',color=C['blue'],ms=5);a.set_xticks(dims.d);a.set_xlabel('Nuclear dimension $d$ (synthetic ladder)');a.set_ylabel('Real-waveform infidelity');a.set_ylim(3e-6,5e-4);a.yaxis.set_minor_formatter(NullFormatter())
save(f,'dimension_transfer',dims,['dimension_transfer.csv'])

pulse_name='robust_N_B1.6_dual_refined';pr,x,_=load_pulse(ROOT,pulse_name);env=envelope_function(pr,x);tt=np.linspace(0,pr.T,601);aa=env(tt)
order=np.argsort(pr.freq);f,a=plt.subplots(figsize=(4.7,3.3),layout='constrained')
im=a.pcolormesh(tt,np.arange(pr.J),1000*np.abs(aa[:,order].T),shading='nearest',cmap='cividis',vmin=0,vmax=50)
ylabels=[f'{pr.freq[k]:.2f} ({"e" if k>=7 else "g"})' for k in order]
a.set_yticks(np.arange(pr.J),ylabels,fontsize=6);a.set_ylabel('Carrier frequency (MHz), electron state');a.set_xlabel('Time ($\\mu$s)')
f.colorbar(im,ax=a,label='Envelope magnitude $|a_j|$ (kHz)',pad=.02)
pddata=pd.DataFrame({'time_us':tt});pddata=pd.concat([pddata,pd.DataFrame(1000*np.abs(aa),columns=[f'tone_{i}_kHz' for i in range(pr.J)])],axis=1)
save(f,'pulse_envelopes',pddata,[pulse_name+'.json','pulses/'+pulse_name+'.npz'])

# tracked-design self-consistency figure (supplement)
trk=[]
for mech,act,lab in cases:
    p=R/f'tracked_{mech}_B1.6_{act}.json'
    if not p.exists():continue
    rec=json.loads(p.read_text())
    for e in rec['log']:
        if 'offsets_kHz' in e:trk.append(dict(label=lab,stage=e['stage'],used_kHz=np.mean(e['offsets_kHz']),updated_kHz=np.mean(e.get('updated_offsets_kHz',e['offsets_kHz'])),u2=e['mean_u2_MHz2']))
if trk:
    tk=pd.DataFrame(trk);f,a=sfig(h=2.3,w=4.0)
    for (lab,g),col,mk_ in zip(tk.groupby('label',sort=False),[C['orange'],C['teal'],C['purple'],C['blue']],['o','s','D','^']):
        a.plot(range(len(g)),-g.updated_kHz,marker=mk_,color=col,label=lab,ms=4)
    a.set_xlabel('Self-consistency round');a.set_ylabel('Predicted comb shift (kHz)');a.legend(loc='upper left',bbox_to_anchor=(1.0,1.0),fontsize=6.3);a.xaxis.set_major_locator(FixedLocator(range(4)))
    save(f,'tracking_convergence',tk,['tracked_*_B1.6_*.json'])

# ================================================================ START TIMES / COMPOSABLE GATES
def load_json(path):return json.loads(path.read_text()) if path.exists() else None
scan_specs=[('nominal_N_B1.35_dual','1.35 T, 14 tones',C['orange'],'o'),('nominal_N_B1.6_dual','1.6 T, 14 tones',C['blue'],'s'),('nominal_nuclear_B1.6_single','1.6 T, 7 tones',C['teal'],'D'),
            ('nominal_N_B1.03_dual','1.03 T, 14 tones',C['purple'],'^'),('nominal_N_B1.714_dual','1.714 T, 14 tones',C['gold'],'v'),('tracked_electron_B1.6_single','1.6 T, 7 tones, $O_E$ tracked',C['ink'],'x'),('robust_N_B1.6_dual_refined','1.6 T, 14 tones, 48 $\\mu$s robust',C['gray'],'+')]
scans={n:load_json(R/f'start_time_{n}.json') for n,*_ in scan_specs}
comp_cases=[('composable_N_B1.6_dual','1.6 T\n14 tones','nominal_N_B1.6_dual'),('composable_N_B1.6_single','1.6 T\n7 tones','nominal_nuclear_B1.6_single'),('composable_N_B1.03_dual','1.03 T\n14 tones','nominal_N_B1.03_dual'),('composable_N_B1.714_dual','1.714 T\n14 tones','nominal_N_B1.714_dual'),('composable_N_B1.35_dual','1.35 T\n14 tones','nominal_N_B1.35_dual')]
comps={n:load_json(R/f'validation_{n}.json') for n,*_ in comp_cases}
windows={n.replace('composable_','window_'):load_json(R/f'validation_{n.replace("composable_","window_")}.json') for n,*_ in comp_cases}
wides={n:load_json(R/f'wide_scan_{n}.json') for n in [c[2] for c in comp_cases]+[c[0].replace('composable_','window_') for c in comp_cases]}
def seq_of(v,idle):
    return [q for q in v['sequences'] if q.get('idle_us')==idle][0]

# main figure: (a) scans of three nominal pulses, (b) composable summary
f=plt.figure(figsize=(COL,5.6),layout='constrained');gs=f.add_gridspec(2,1,height_ratios=[1,1.45])
a=f.add_subplot(gs[0]);grid(a);scanrows=[]
for n,lab,col,mk_ in scan_specs[:3]:
    sc=scans.get(n)
    if not sc:continue
    g=sorted(sc['scan'],key=lambda z:z['t0_us']);t=[z['t0_us'] for z in g];fv=[1-z['worst'] for z in g]
    a.semilogy(t,np.maximum(fv,1e-6),marker=mk_,color=col,ms=3.6,lw=1,label=lab)
    for z in g:scanrows.append(dict(pulse=n,t0_us=z['t0_us'],worst=z['worst']))
a.axhline(1e-2,color=C['gray'],ls='--',lw=.7);a.text(.1,1.3e-2,'1% error',ha='left',va='bottom',fontsize=6,color=C['gray'])
a.set_xlabel('Pulse start time $t_0$ ($\\mu$s), carriers running continuously');a.set_ylabel('Gate infidelity $1-F$');a.set_ylim(5e-6,1);a.set_xlim(-.2,10.2);a.yaxis.set_minor_formatter(NullFormatter())
a.legend(fontsize=6,loc='center right',bbox_to_anchor=(1.0,.47),handlelength=1.3);label(a,'(a)',x=-.2,y=1.0)
b=f.add_subplot(gs[1]);grid(b);xx=np.arange(len(comp_cases));comprows=[]
for i,(n,lab,nomn) in enumerate(comp_cases):
    v=comps.get(n);wd=wides.get(nomn)
    if wd:
        b.plot(i-.3,1-wd['worst_point_L2']['worst'],'o',color=C['gray'],ms=4.5,zorder=3)
    if v:
        held=[1-z['worst'] for z in v['held_out']];med=float(np.median(held))
        b.plot([i-.1,i-.1],[min(held),max(held)],color=C['blue'],lw=1.6,solid_capstyle='round',zorder=2);b.plot(i-.1,med,'s',color=C['blue'],ms=4.5,zorder=3)
        b.plot(i+.05,1-v['design_mesh_scan']['min'],'v',color=C['purple'],ms=4.5,zorder=3)
        b.plot(i+.2,1-v['full_electron'][1]['worst'],'D',color=C['teal'],ms=4.5,zorder=3)
        seq=seq_of(v,0.);b.plot(i+.35,1-seq['full_worst'],'^',color=C['orange'],ms=4.5,zorder=3)
        s3=seq_of(v,'0.73,2.38');b.plot(i+.5,1-s3['worst'],'P',color=C['gold'],ms=4.8,zorder=3)
        comprows.append(dict(name=n,held_out_min=v['held_out_min'],held_out_median=v['held_out_median'],scan_min_L1=v['design_mesh_scan']['min'],full_worst_start=v['full_electron'][1]['worst'],two_pulse_full=seq['full_worst'],three_pulse=s3['worst'],nominal_worst_wide_scan_L2=wd['worst_point_L2']['worst'] if wd else np.nan))
b.plot([],[],'o',color=C['gray'],label='nominal pulse, worst of 400 start times in 0$-$100 ms (L2)',ms=4);b.plot([],[],'s',color=C['blue'],label='composable pulse, 48 held-out start times (L2)',ms=4)
b.plot([],[],'v',color=C['purple'],label='composable, worst of 400 scanned start times (L1)',ms=4)
b.plot([],[],'D',color=C['teal'],label='composable, worst held-out start time (L4)',ms=4);b.plot([],[],'^',color=C['orange'],label='composable, two pulses vs $G^2$ (L4)',ms=4);b.plot([],[],'P',color=C['gold'],label='composable, three pulses vs $G^3$ (L2)',ms=4)
b.axhline(1e-2,color=C['gray'],ls='--',lw=.7)
b.set_xticks(xx,[lab for _,lab,_ in comp_cases],fontsize=6.3);b.set_ylabel('Gate infidelity $1-F$');b.set_yscale('log');b.set_ylim(3e-6,1);b.set_xlim(-.6,4.7);b.yaxis.set_minor_formatter(NullFormatter())
b.legend(fontsize=5.8,loc='upper left',bbox_to_anchor=(-0.02,-0.2),handlelength=1.2,frameon=False,ncol=1);label(b,'(b)',x=-.2,y=1.0)
save(f,'fig_composable',pd.DataFrame(comprows),['start_time_*.json','validation_composable_*.json','wide_scan_*.json']);pd.DataFrame(scanrows).to_csv(D/'fig_composable_scans.csv',index=False)
# supplement figure: all scans
f,a=sfig(h=2.8,w=5.0);allscan=[]
for n,lab,col,mk_ in scan_specs:
    sc=scans.get(n)
    if not sc:continue
    g=sorted(sc['scan'],key=lambda z:z['t0_us']);a.semilogy([z['t0_us'] for z in g],np.maximum([1-z['worst'] for z in g],1e-7),marker=mk_,color=col,ms=3.2,lw=.9,label=lab)
    for z in g:allscan.append(dict(pulse=n,t0_us=z['t0_us'],worst=z['worst']))
a.axhline(1e-2,color=C['gray'],ls='--',lw=.7);a.set_xlabel('Pulse start time $t_0$ ($\\mu$s)');a.set_ylabel('Gate infidelity $1-F$');a.set_ylim(5e-8,1);a.yaxis.set_minor_formatter(NullFormatter())
a.legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.0,1.0));save(f,'start_time_scans',pd.DataFrame(allscan),['start_time_*.json'])
# supplement figure: wide-range scans (L1) of nominal, window-trained and composable pulses at 1.6 T and 1.35 T
f,axs=plt.subplots(2,1,figsize=(5.0,4.2),layout='constrained',sharex=True);widerows=[]
for a_,(nomn,winn,compn,ttl) in zip(axs,[('nominal_N_B1.6_dual','window_N_B1.6_dual','composable_N_B1.6_dual','1.6 T, 14 tones'),('nominal_N_B1.35_dual','window_N_B1.35_dual','composable_N_B1.35_dual','1.35 T, 14 tones')]):
    grid(a_)
    for name_,lab_,col_,mk_ in ((nomn,'nominal',C['gray'],'o'),(winn,'trained on 0$-$10 $\\mu$s',C['orange'],'s'),(compn,'trained on 0$-$100 ms (composable)',C['blue'],'D')):
        rec=wides.get(name_) if name_ in wides else comps.get(name_)
        if not rec:continue
        pts=rec['points'] if 'points' in rec else rec['design_mesh_scan']['points']
        t=np.array([z['t0_us'] for z in pts])/1e3;fv=np.maximum([1-z['worst'] for z in pts],1e-7)
        a_.semilogy(t,fv,mk_,color=col_,ms=2.6,alpha=.75,label=lab_,mew=0);a_.axhline(fv.max(),color=col_,lw=.8,ls=':')
        for z in pts:widerows.append(dict(pulse=name_,t0_us=z['t0_us'],worst=z['worst']))
    a_.set_ylabel('Gate infidelity $1-F$');a_.set_ylim(1e-6,1);a_.yaxis.set_minor_formatter(NullFormatter())
    a_.text(.01,.97 if ttl.startswith('1.6') else .04,ttl,transform=a_.transAxes,ha='left',va='top' if ttl.startswith('1.6') else 'bottom',fontsize=7)
axs[0].legend(fontsize=6,loc='lower right',ncol=3,frameon=True,framealpha=.95,edgecolor='none');axs[1].set_xlabel('Pulse start time $t_0$ (ms), carriers running continuously')
save(f,'wide_scan',pd.DataFrame(widerows),['wide_scan_*.json','validation_composable_*.json'])
# tables
strows=[]
for n,lab,*_ in scan_specs:
    sc=scans.get(n)
    if not sc:continue
    ws=[z['worst'] for z in sc['scan']];fullw=sc['full'][0]['worst'] if sc['full'] else None
    seq=sc['sequences'][0] if sc['sequences'] else None
    strows.append([lab.replace('$\\mu$s','$\\mu$s'),pct(ws[0],4),pct(min(ws),3),pct(float(np.median(ws)),3),pct(fullw,3) if fullw else '--',pct(seq['full_worst'],3) if seq and seq.get('full_worst') else '--'])
table('starttime_table','Start-time dependence of the archived pulses (24 start times in $[0,10]\\,\\mu$s): fidelity at $t_0=0$, minimum and median over start times (L2; L3 for the tracked pulse), full-model fidelity at the worst start time (L4), and two-pulse sequence with zero idle against $G^2$ (L4).', ['Pulse','$t_0=0$','Minimum','Median','Worst, full','Two pulses, full'],strows,'lrrrrr',label='tab:starttime')
wrows=[]
for n,lab,nomn in comp_cases:
    wn=n.replace('composable_','window_');wv=windows.get(wn);ww=wides.get(wn);nw=wides.get(nomn)
    if not (wv and ww):continue
    wrows.append([lab.replace('\n',', '),pct(wv['held_out_min'],3),pct(ww['audit_points'][0]['worst'],3),pct(ww['audit_points'][1]['worst'],3),pct(ww['scan_min'],3),pct(ww['worst_point_L2']['worst'],3),pct(ww['worst_point_full']['worst'],3),pct(nw['worst_point_L2']['worst'],3) if nw else '--'])
table('window_table','Pulses trained on eight start times within $[0,10]\\,\\mu$s (window-trained): minimum over 24 held-out start times in the same window (L2); fidelity at the two start times of the external audit, $43.417$ and $769.616\\,\\mu$s (L2); minimum of a 400-point scan over $[0,100]$~ms on the design mesh (L1) and the worst scanned point re-propagated at L2 and L4; and, for comparison, the worst scanned point of the nominal pulse over the same range (L2).', ['Pulse','Held-out min (window)','$43.4\\,\\mu$s','$769.6\\,\\mu$s','Scan min (L1)','Worst (L2)','Worst (L4)','Nominal, worst (L2)'],wrows,'lrrrrrrr',label='tab:window',size=r'\footnotesize',colsep='3pt',resize=True)
crows=[]
comps['refined_N_B1.6_dual']=load_json(R/'validation_refined_N_B1.6_dual.json')
for n,lab,nomn in comp_cases+[('refined_N_B1.6_dual','1.6 T, 14 tones (24 start times)','nominal_N_B1.6_dual')]:
    v=comps.get(n)
    if not v:continue
    seq0=seq_of(v,0.);seq1=seq_of(v,1.37);seqL=seq_of(v,1000.37);seq3=seq_of(v,'0.73,2.38');seq3w=seq_of(v,'87.3,1926.4')
    crows.append([lab.replace('\n',', '),pct(v['full_electron'][0]['worst'],4),pct(v['held_out_min'],3),pct(v['held_out_median'],3),pct(v['design_mesh_scan']['min'],3),pct(v['full_electron'][1]['worst'],3),pct(seq0['full_worst'],3),pct(seq1['worst'],3),pct(seqL['worst'],3),pct(seq3['worst'],3),pct(seq3w['worst'],3)])
table('composable_table','Composable pulses trained over sixteen start times drawn from $[0,100]$~ms (last row: the $1.6$~T fourteen-tone case re-run with 24 start times and up to 1200 iterations from the composable pulse): full-model fidelity at $t_0=0$ (L4); minimum and median over 48 held-out start times from the same range (L2); minimum of a 400-point scan on the design mesh (L1); full-model fidelity at the worst held-out start time (L4); two pulses with zero idle (L4), with a $1.37$-$\\mu$s idle and with a $1000.37$-$\\mu$s idle (L2); three pulses with consecutive idles of $0.73$ and $2.38\\,\\mu$s and with consecutive idles of $87.3$ and $1926.4\\,\\mu$s (L2), each against the product of the ideal gates.', ['Pulse','$t_0=0$ (L4)','Min.','Median','Scan (L1)','Worst (L4)','2, idle 0 (L4)','2, 1.37','2, 1000','3 pulses','3, long'],crows,'lrrrrrrrrrr',label='tab:composable',size=r'\footnotesize',colsep='3pt',resize=True)
trk_rob={n:load_json(R/f'tracked_robustness_{n}.json') for n in ('tracked_electron_B1.6_single','tracked_electron_B1.6_dual')}
trrows=[]
for n,lab in (('tracked_electron_B1.6_single','$O_E$, 7 tones'),('tracked_electron_B1.6_dual','$O_E$, 14 tones')):
    r_=trk_rob.get(n)
    if not r_:continue
    gm={g['gain']:g['worst'] for g in r_['gain']};ax={a_['axis']:a_['worst'] for a_ in r_['axes']}
    trrows.append([lab,pct(gm[1.0],4),pct(gm[0.97],2),pct(gm[1.03],2),pct(min(ax['B-'],ax['B+']),3),pct(min(ax['k-'],ax['k+'],ax['A-'],ax['A+']),3),pct(r_['worst_corner']['worst'],2),pct(r_['worst_corner_full']['worst'],2) if r_.get('worst_corner_full') else '--'])
table('tracked_robustness_table','Frozen shift-tracked $O_E$ pulses over the residual-error domain (L3 unless stated): nominal, $\\pm3\\%$ gain, worst of the field endpoints, worst of the hyperfine/quadrupole endpoints, worst of the sixteen simultaneous corners, and the worst corner in the full 16-level model (L4).', ['Pulse','Nominal','$-3\\%$ gain','$+3\\%$ gain','$\\pm0.1$ mT','$\\pm2$ kHz','Worst corner','Worst corner, full'],trrows,'lrrrrrrr',label='tab:trackedrobust')

# ================================================================ TABLES
allnom=[];resrows=[]
for p in sorted(R.glob('validation_nominal*.json'))+sorted(R.glob('validation_selective*.json')):
    v=json.loads(p.read_text());n=v['name'];cfg=j(n)['config'];mech=cfg.get('mechanism','nuclear');act=len(cfg['active']);b_=cfg['B'];kind='selective' if n.startswith('selective') else 'rotation'
    lab={'nuclear':'$O_N$','electron':'$O_E$','magnetic':'$O_M$'}[mech]+f', {b_:g} T, {7*act} tones'+(' (selective)' if kind=='selective' else '')
    allnom.append([lab,f'{cfg["T"]:g}',pct(v["physical_projected"]["worst"],5),pct(v["full_electron"][-1]["worst"],5),pct(v["plain_multitone"]["worst"],4) if v.get('plain_multitone') else '--'])
    r0=v['resources'];resrows.append([lab,f'{1000*r0["peak_tone_MHz"]:.1f}',f'{1000*r0["peak_rss_MHz"]:.1f}',f'{r0["fluence_MHz2_us"]:.4f}',f'{r0["envelope_99pct_power_halfband_MHz"]:.3f}'])
table('nominal_table','Independent tests of the nominal gates. \\emph{Projected} is the adaptive real-waveform propagation in the projected branches; \\emph{Full} is the propagation of the complete driven 16-level Hamiltonian with leakage counted as error; \\emph{Unshaped} is the fixed-amplitude multitone pulse propagated in the projected model. The last row is the selective rotation.', ['Control','$T$ ($\\mu$s)','Projected','Full','Unshaped'],allnom,'lrrrr',label='tab:nominal')
table('resource_table','Resources of each nominal waveform on a 1-ns grid: peak tone amplitude and peak root-sum-square envelope (kHz), envelope fluence (MHz$^2\\,\\mu$s), and the half-band containing 99\\% of the baseband envelope power (MHz).', ['Control','$a_{\\rm peak}$','RSS peak','Fluence','Half-band'],resrows,'lrrrr',label='tab:resources')
st=[]
for mech,act,lab in cases:
    old=R/f'validation_stark_{mech}_B1.6_{act}.json';new=R/f'validation_tracked_{mech}_B1.6_{act}.json'
    lin=json.loads((R/f'validation_nominal_{mech}_B1.6_{act}.json').read_text())
    o=json.loads(old.read_text()) if old.exists() else None;nw=json.loads(new.read_text()) if new.exists() else None
    st.append([lab,pct(lin['physical_projected']['worst'],4),pct(lin['full_electron'][-1]['worst'],2),pct(o['second_order_physical']['worst'],2) if o else '--',
               f'{-np.mean(nw["carrier_offset_kHz"]):.1f}' if nw else '--',pct(nw['second_order_physical']['worst'],4) if nw else '--',pct(nw['full_electron'][-1]['worst'],4) if nw else '--'])
table('stark_table','Electron-coupled drives at 1.6 T, $12\\,\\mu$s. Columns: design with the linearly projected operator evaluated in its own model and in the full 16-level model; the same linear design re-optimized with the second-order term but with unshifted carriers and the unshifted target (\\emph{untracked}, evaluated in the second-order model, L3; the full-model values differ from these by less than $0.1\\%$); and the tracked design (carriers moved by the listed mean comb shift, frame defined by the carriers), evaluated with the second-order model (L3) and with the full 16-level Hamiltonian (L4). Column labels give the validation level of Table~I of the main text.', ['Case','Linear (L2)','Linear (L4)','Untracked (L3)','Shift (kHz)','Tracked (L3)','Tracked (L4)'],st,'lrrrrrr',label='tab:stark',size=r'\footnotesize')
rt=[]
for n in ['nominal_N_B1.6_dual','nominal_nuclear_B1.6_single','robust_N_B1.6_dual_seed1','robust_N_B1.6_dual_seed2','robust_N_B1.6_dual_seed3','robust_N_B1.6_single_seed1','robust_N_B1.6_dual_corners','robust_N_B1.6_dual_refined']:
    p=R/('robust_validation_'+n+'.json')
    if not p.exists():continue
    v=json.loads(p.read_text());cfg=j(n)['config']
    lab={'nominal_N_B1.6_dual':'Nominal, 14 tones','nominal_nuclear_B1.6_single':'Nominal, 7 tones','robust_N_B1.6_dual_corners':'Full corners, 14 tones','robust_N_B1.6_dual_refined':'Refined, 14 tones','robust_N_B1.6_single_seed1':'Axis start 1, 7 tones'}.get(n,'Axis start '+n[-1]+', 14 tones')
    rt.append([lab,f'{cfg["T"]:g}',pct(v['nominal_physical']['worst'],3),pct(v["corner_rwa_min"],3),pct(v["heldout_rwa_min"],3),pct(v["selected_corner_physical"]["worst"],3)])
table('robust_table','Calibration tests at 1.6 T over the residual-error domain of the main text. \\emph{Nominal} is the real-waveform fidelity at the design point; corner and held-out minima use the independent finer-mesh rotating-wave evaluation; the last column is the adaptive real-waveform result at the corner selected as worst. None is a certificate over the continuous domain.', ['Control','$T$ ($\\mu$s)','Nominal','Worst corner','Worst held-out','Worst corner (real)'],rt,'lrrrrr',label='tab:robust')
nt=[[f'{row.p_exc:.5g}',f'{row.tau_c_us:g}',f'{row.fidelity:.7f}',f'{row.sem:.2g}',f'{row.nojump_probability:.5g}'] for row in noise.itertuples()]
table('noise_table','Prepared-lower-state electron channel at 1.6 T and $12\\,\\mu$s. Each point uses 8192 paths conditional on at least one jump; the no-jump contribution is exact within the propagated model. The uncertainty is the Monte Carlo standard error.', ['$p_e$','$\\tau_c$ ($\\mu$s)','$F$','Standard error','$P_0$'],nt,label='tab:noise')
reqr=[[f'{row.B_T:g}',f'{row.T_us:g}',f'{row.p_exc_sufficient:.3g}',f'{row.temperature_sufficient_mK:.1f}',f'{row.motional_tau_fs:.3g}'] for row in rr.itertuples()]
table('requirements_table','Sufficient polarization bound with $\\tau_c=0.1\\,\\mu$s, total error $1\\%$, and coherent-error allowance $0.1\\%$; thermal-equivalent temperature for the nominal electron splitting; and the fast-fluctuation (motional-narrowing) correlation time that would be needed at the $0.4$-K population.', ['$B$ (T)','$T$ ($\\mu$s)','$p_e$ bound','$T_e$ (mK)','$\\tau_{\\rm MN}$ (fs)'],reqr,label='tab:req')
table('dimension_table','Synthetic dimension-transfer tests with the same conservative total-envelope bound. The ladders are not assigned to physical hosts.', ['$d$','$I$','Tones','$a_{\\max}$ (kHz)','Real-waveform fidelity'],[[str(int(z.d)),f'{z.I:g}',str(int(z.J)),f'{1000*z.amax_MHz:.2f}',pct(z.physical_fidelity,5)] for z in dims.itertuples()],label='tab:dim')
if (R/'diagonal_output_diagnostic.json').exists():
    dr=[]
    for v in j('diagonal_output_diagnostic'):
        if 'best_diagonal_output_fidelities' not in v:continue
        lab=('$O_E$' if 'electron' in v['name'] else '$O_M$')+(', 7 tones' if 'single' in v['name'] else ', 14 tones')
        dr.append([lab,pct(v["worst"],2),pct(min(v["best_diagonal_output_fidelities"]),2)])
    table('diagonal_table','Diagnostic for the linearly designed electron-coupled pulses: fidelity to the fixed target and the best fidelity attainable by a final diagonal phase correction fitted to the simulated output. The fitted correction is not used anywhere else.', ['Control','Fixed target','Best final diagonal'],dr,label='tab:diag')
table('source_uncertainty_table','Alternative readings of the source quadrupole uncertainty and of the nuclear $g$ factor, and the resulting range of the field at which the two combs are maximally separated ($r=1$ maximum). The intervals combine endpoint ranges and are not statistical confidence intervals.', ['$\\sigma_{\\kappa47}$ (MHz)','Nuclear $g$ factor','$\\sigma_{\\kappa49}$ (MHz)','$B_{r=1}$ range (T)'],[[f'{z.sigma_k47_MHz:g}','bulk ($0.315$)' if z.source_gamma=='bulk_coefficient' else 'surface fit ($0.37\\pm0.04$)',f'{z.sigma_k49_MHz:.2f}',f'{z.B_lower_T:.3f}--{z.B_upper_T:.3f}'] for z in unc.itertuples()],'rlrr',label='tab:unc')

# ================================================================ MACROS
ng=noise[(np.isclose(noise.p_exc,pt))&(np.isclose(noise.tau_c_us,.1))].iloc[0]
slow=noise[(np.isclose(noise.p_exc,pt))&(np.isclose(noise.tau_c_us,1000))].iloc[0]
pol=noise[(np.isclose(noise.p_exc,1e-4,rtol=1e-10,atol=0))&(np.isclose(noise.tau_c_us,.1))].iloc[0]
def trackedpct(mech,act):
    p=R/f'validation_tracked_{mech}_B1.6_{act}.json'
    return pct(json.loads(p.read_text())['full_electron'][-1]['worst'],3) if p.exists() else '(pending)'
def scanmin(n,digits=1):
    sc=scans.get(n);return pct(min(z['worst'] for z in sc['scan']),digits) if sc else '(pending)'
def seqfull(n,digits=1):
    sc=scans.get(n);return pct(sc['sequences'][0]['full_worst'],digits) if sc and sc['sequences'] and sc['sequences'][0].get('full_worst') else '(pending)'
def transfer(n):
    sc=scans.get(n);tp=sc['sequences'][0].get('transfer_probability') if sc and sc['sequences'] else None
    return ' and '.join(f'{100*x:.0f}\\%' for x in tp) if tp else '(pending)'
def comppct(n,key,digits=2):
    v=comps.get(n)
    if not v:return '(pending)'
    val={'held':v['held_out_min'],'full':v['full_electron'][1]['worst'],'scan':v['design_mesh_scan']['worst_point_L2']['worst'],
         'two':seq_of(v,0.)['full_worst'],'three':seq_of(v,'0.73,2.38')['worst'],
         'seqmin':min([q.get('full_worst',q['worst']) for q in v['sequences']]+[q['worst'] for q in v['sequences']])}[key];return pctfloor(val,digits)
def widepct(n,key,digits=3):
    w=wides.get(n)
    if not w:return '(pending)'
    val={'scan':w['worst_point_L2']['worst'],'audit':w['audit_points'][1]['worst']}[key];return pct(val,digits)
def windowheld(n,digits=3):
    v=windows.get(n);return pct(v['held_out_min'],digits) if v else '(pending)'
def resolved_min(key,digits=2):
    # minimum over the composable pulses at the fields where the combs are resolved (all cases except 1.35 T)
    vals=[]
    for n in ('composable_N_B1.6_dual','composable_N_B1.6_single','composable_N_B1.03_dual','composable_N_B1.714_dual'):
        v=comps.get(n)
        if not v:return '(pending)'
        if key=='held':vals+=[v['held_out_min'],v['design_mesh_scan']['worst_point_L2']['worst'],v['full_electron'][1]['worst']]
        else:vals+=[q.get('full_worst',q['worst']) for q in v['sequences']]+[q['worst'] for q in v['sequences']]
    return pctfloor(min(vals),digits)
def trk(n,what):
    r_=trk_rob.get(n)
    if not r_:return '(pending)'
    if what=='gain':gm={g['gain']:g['worst'] for g in r_['gain']};return f"{100*gm[0.97]:.1f}\\% and {100*gm[1.03]:.1f}\\%"
    if what=='corner':return pct(r_['worst_corner']['worst'],1)
    if what=='cornerfull':return pct(r_['worst_corner_full']['worst'],1) if r_.get('worst_corner_full') else '(pending)'
from qudit_control.requirements import maximum_excited_fraction
robust_eps=1-rob[rob.name=='robust_N_B1.6_dual_refined'].corner.iloc[0]
pe_robust=maximum_excited_fraction(48.,.1,total_error=.01,coherent_error=float(robust_eps))
macros={'SingleNPercent':pct(nom.iloc[-1].optimized_full_fidelity,4),'PlainSinglePercent':pct(nom.iloc[-1].plain_projected_fidelity,4),
        'StartMinThirtyFivePercent':scanmin('nominal_N_B1.35_dual',0),'StartSeqThirtyFivePercent':seqfull('nominal_N_B1.35_dual',0),'StartTransferThirtyFive':transfer('nominal_N_B1.35_dual'),
        'StartMinSixteenPercent':scanmin('nominal_N_B1.6_dual',3),'StartMinSinglePercent':scanmin('nominal_nuclear_B1.6_single',3),'StartMinTrackedPercent':scanmin('tracked_electron_B1.6_single',3),
        'CompSixteenHeldPercent':comppct('composable_N_B1.6_dual','held',3),'CompSixteenFullPercent':comppct('composable_N_B1.6_dual','full',3),'CompSeventeenHeldPercent':comppct('composable_N_B1.714_dual','held',3),
        'CompOneThreeHeldPercent':comppct('composable_N_B1.03_dual','held',3),'CompSingleHeldPercent':comppct('composable_N_B1.6_single','held',3),'CompThirtyFiveHeldPercent':comppct('composable_N_B1.35_dual','held',1),
        'CompSixteenScanPercent':comppct('composable_N_B1.6_dual','scan',3),'CompSixteenTwoPercent':comppct('composable_N_B1.6_dual','two',3),'CompSixteenThreePercent':comppct('composable_N_B1.6_dual','three',3),'CompThirtyFiveThreePercent':comppct('composable_N_B1.35_dual','three',1),
        'CompResolvedHeldMinPercent':resolved_min('held',1),'CompResolvedSeqMinPercent':resolved_min('seq',1),'CompSixteenSeqMinPercent':comppct('composable_N_B1.6_dual','seqmin',1),'NominalSixteenWidePercent':widepct('nominal_N_B1.6_dual','scan',2),'WindowSixteenHeldPercent':windowheld('window_N_B1.6_dual',3),'WindowSixteenAuditPercent':widepct('window_N_B1.6_dual','audit',2),'WindowSixteenWidePercent':widepct('window_N_B1.6_dual','scan',2),
        'TrackedGainSinglePercent':trk('tracked_electron_B1.6_single','gain'),'TrackedGainDualPercent':trk('tracked_electron_B1.6_dual','gain'),'TrackedCornerSinglePercent':trk('tracked_electron_B1.6_single','corner'),'TrackedCornerSingleFullPercent':trk('tracked_electron_B1.6_single','cornerfull'),
        'PeRequirementRobust':f'{pe_robust:.1e}'.replace('e-0','\\times10^{-').replace('e-','\\times10^{-')+'}',
        'ThermalNoiseF':f'{ng.fidelity:.3f}','SlowNoiseF':f'{slow.fidelity:.5f}','PolarizedNoiseF':f'{pol.fidelity:.4f}',
        'TrackedElectronSinglePercent':trackedpct('electron','single'),'TrackedElectronDualPercent':trackedpct('electron','dual'),
        'TrackedMagneticSinglePercent':trackedpct('magnetic','single'),'TrackedMagneticDualPercent':trackedpct('magnetic','dual'),
        'SlopeExact':f'{slope_exact:.1f}','SlopeAnalytic':f'{slope_analytic:.1f}','PeRequirement':f'{pe_req:.2e}'}
(ROOT/'manuscript/generated/numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
pd.DataFrame(manifest).to_csv(ROOT/'provenance/figure_manifest.csv',index=False)
key=[dict(metric='nominal_full_14tone_1.6T',value=float(nom.iloc[2].optimized_full_fidelity),unit='fidelity'),dict(metric='nominal_full_7tone_1.6T',value=float(nom.iloc[-1].optimized_full_fidelity),unit='fidelity'),
     dict(metric='refined_selected_corner',value=float(rob.iloc[-1].corner),unit='fidelity'),dict(metric='prepared_thermal_0.1us',value=float(ng.fidelity),unit='fidelity'),dict(metric='prepared_p1e-4_0.1us',value=float(pol.fidelity),unit='fidelity'),dict(metric='sufficient_pe_12us',value=pe_req,unit='probability'),
     dict(metric='comb_shift_slope_exact_MHz_per_MHz2',value=float(slope_exact),unit='MHz/MHz^2'),dict(metric='comb_shift_slope_analytic_MHz_per_MHz2',value=float(slope_analytic),unit='MHz/MHz^2')]
for row in op.itertuples():key.append(dict(metric=f'tracked_full_{row.mechanism}_{row.tones}tones',value=float(row.tracked_full),unit='fidelity'))
pd.DataFrame(key).to_csv(R/'key_results_summary.csv',index=False)
print(f'Built {len(manifest)} figures, data tables, LaTeX tables and macros.')
