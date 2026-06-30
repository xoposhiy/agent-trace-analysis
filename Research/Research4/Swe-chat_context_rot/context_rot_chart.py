"""Generate the context-rot cost chart (PNG) from local conversations.parquet.

Two panels:
  (left)  avg $/turn by per-turn context size, split into the cache-read
          "context tax" vs other cost -> shows total rising and tax dominating.
  (right) share of turns by context-window occupancy -> shows the window rarely
          fills (almost nothing reaches 1M).
"""
import glob, os
from collections import defaultdict
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--SALT-NLP--SWE-chat/snapshots/*/conversations.parquet"))[0]

def rate(m):
    m = (m or "").lower()
    for k, r in [("fable-5",(10,50)),("opus-4-8",(5,25)),("opus-4-7",(5,25)),("opus-4-6",(5,25)),
                 ("opus-4-5",(5,25)),("opus-4-1",(15,75)),("opus-4-20",(15,75)),("opus",(15,75)),
                 ("sonnet",(3,15)),("haiku-4",(1,5)),("3-5-haiku",(.8,4)),("haiku",(1,5))]:
        if k in m: return r
    return (3,15)

t = pq.read_table(P, columns=["model","input_tokens","output_tokens",
                              "cache_creation_input_tokens","cache_read_input_tokens"])
d = t.to_pydict()

BINS = [(0,25_000),(25_000,50_000),(50_000,100_000),(100_000,150_000),
        (150_000,200_000),(200_000,300_000),(300_000,500_000),(500_000,10**12)]
LABELS = ["0-25k","25-50k","50-100k","100-150k","150-200k","200-300k","300-500k",">500k"]
cr_sum = defaultdict(float); other_sum = defaultdict(float); cnt = defaultdict(int)
ctxs = []

for i in range(t.num_rows):
    inp=d["input_tokens"][i] or 0; out=d["output_tokens"][i] or 0
    cw=d["cache_creation_input_tokens"][i] or 0; cr=d["cache_read_input_tokens"][i] or 0
    if inp+out+cw+cr==0: continue
    ir,orr=rate(d["model"][i])
    crc=cr/1e6*ir*0.10
    other=(inp/1e6*ir)+(out/1e6*orr)+(cw/1e6*ir*1.25)
    ctx=inp+cw+cr; ctxs.append(ctx)
    for b in BINS:
        if b[0] <= ctx < b[1]:
            cr_sum[b]+=crc; other_sum[b]+=other; cnt[b]+=1; break

avg_cr = [cr_sum[b]/cnt[b] if cnt[b] else 0 for b in BINS]
avg_other = [other_sum[b]/cnt[b] if cnt[b] else 0 for b in BINS]

N=len(ctxs)
THR=[100_000,250_000,500_000,750_000,1_000_000]
share=[100*sum(1 for c in ctxs if c>=th)/N for th in THR]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.2))

x=range(len(BINS))
ax1.bar(x,avg_other,label="other (input+output+cache-write)",color="#9ecae1")
ax1.bar(x,avg_cr,bottom=avg_other,label="cache-read (context tax)",color="#d62728")
ax1.set_xticks(list(x)); ax1.set_xticklabels(LABELS,rotation=40,ha="right",fontsize=9)
ax1.set_xlabel("per-turn context size (tokens)")
ax1.set_ylabel("avg cost per turn ($)")
ax1.set_title("Cost per turn rises with context\n(red = money spent re-reading context)")
ax1.legend(fontsize=8,loc="upper left")
for xi,(o,c) in enumerate(zip(avg_other,avg_cr)):
    ax1.text(xi,o+c,f"${o+c:.2f}",ha="center",va="bottom",fontsize=8)

ax2.bar([f"≥{t//1000}k" for t in THR],share,color="#756bb1")
ax2.set_ylabel("% of turns")
ax2.set_xlabel("per-turn context-window occupancy")
ax2.set_title("Window rarely fills\n(~0% of turns reach 1M)")
for xi,s in enumerate(share):
    ax2.text(xi,s,f"{s:.1f}%",ha="center",va="bottom",fontsize=9)

fig.suptitle("Context rot in dollars — SWE-chat (5,851 sessions, Opus 4.x rates)",
             fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
out=os.path.join(os.path.dirname(__file__),"context_rot_curve.png")
fig.savefig(out,dpi=130)
print("saved",out)
