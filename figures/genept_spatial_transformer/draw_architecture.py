"""Editable publication-style schematic; standard library only.

Run this file, then render architecture.svg with rsvg-convert.
The diagram describes optimizations/scs_streaming/torch_model.py (scale=4).
"""
from pathlib import Path
from html import escape
import math
import random

OUT = Path(__file__).resolve().parent
W, H = 1800, 1210
C = dict(ink="#26343D", muted="#65747C", line="#BBC6CC", blue="#527FAD",
         bluebg="#EAF1F8", teal="#388E87", tealbg="#E8F3F0", orange="#C88955",
         orangebg="#FBF0E6", grey="#F5F7F8", purple="#8A7AA2")
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
     '<title>GenePT-informed spatial transformer for spot-level segmentation</title>',
     '<desc>All-gene sparse expression is encoded independently for each spot using frozen GenePT vectors, expression-weighted mean pooling and a trainable 1536 to 256 projection. Relative spatial positions are added. Fifty tokens enter a 32-layer transformer, followed by centre-spot direction and foreground heads.</desc>',
     '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="#65747C"/></marker></defs>',
     '<rect width="1800" height="1210" fill="white"/>',
     '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#26343D">']

def text(x, y, value, size=20, fill=None, weight=None, anchor=None, italic=False):
    attrs = f'x="{x}" y="{y}" font-size="{size}"'
    if fill: attrs += f' fill="{fill}"'
    if weight: attrs += f' font-weight="{weight}"'
    if anchor: attrs += f' text-anchor="{anchor}"'
    if italic: attrs += ' font-style="italic"'
    s.append(f'<text {attrs}>{escape(str(value))}</text>')

def rect(x,y,w,h,fill="white",stroke=None,r=8,sw=1.3):
    s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def line(x1,y1,x2,y2,arrow=False,color=None,dash=None,sw=1.6):
    extra = ' marker-end="url(#arrow)"' if arrow else ''
    if dash: extra += f' stroke-dasharray="{dash}"'
    s.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color or C["muted"]}" stroke-width="{sw}"{extra}/>')

def path(d,color=None,arrow=False,dash=None):
    extra = ' marker-end="url(#arrow)"' if arrow else ''
    if dash: extra += f' stroke-dasharray="{dash}"'
    s.append(f'<path d="{d}" fill="none" stroke="{color or C["muted"]}" stroke-width="1.6"{extra}/>')

def circle(x,y,r,fill="white",stroke=None,sw=1.5):
    s.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def panel(letter,title,y):
    text(48,y,letter,28,weight="bold")
    text(88,y,title,25,weight="bold")

def vector(x,y,n=12,cell=12,h=27,color=None,seed=1):
    rng=random.Random(seed)
    for i in range(n):
        rect(x+i*(cell+2),y,cell,h,color or C['teal'],r=1)
        s.append(f'<rect x="{x+i*(cell+2)}" y="{y}" width="{cell}" height="{h}" fill="white" opacity="{rng.uniform(.03,.75):.2f}"/>')

text(48,48,"GenePT-informed spatial transformer",32,weight="bold")
text(48,79,"Whole-slide learning with spot-resolved inputs and predictions",19,C['muted'])
rect(1464,26,284,46,C['grey'],r=23)
text(1606,56,"4× width + depth · 22.5M",18,anchor="middle")
line(48,98,1752,98,color=C['line'],sw=1)

# a — sampling / data contract
panel('a','Whole-slide sampling without spatial aggregation',137)
text(169,177,"Whole slide",20,weight="bold",anchor="middle")
rng=random.Random(8)
for row in range(7):
    for col in range(12):
        if ((col-5.5)/6.2)**2+((row-3)/3.8)**2 < 1:
            color = ["#DCE8E5", "#BCD3CF", "#9FBDB8", "#D9E2E7"][rng.randrange(4)]
            rect(62+col*18,190+row*15,16,13,color,r=1)
rect(166,218,39,33,'none',C['orange'],r=1,sw=2.6)
text(169,324,"267 tiles · one shared model",17,C['muted'],anchor="middle")
line(305,239,379,239,arrow=True)
text(341,219,"sample",15,C['muted'],anchor="middle")

for i in range(50):
    if i == 0: continue
    a=i*2.399963
    r=11*math.sqrt(i)
    circle(485+math.cos(a)*r,239+math.sin(a)*r*.72,4.5,C['blue'],sw=0)
circle(485,239,7,C['orange'],C['ink'],sw=1)
s.append('<ellipse cx="485" cy="239" rx="90" ry="69" fill="none" stroke="#BBC6CC" stroke-dasharray="4 5"/>')
text(485,324,"1 centre + 49 neighbouring spots",17,C['muted'],anchor="middle")
line(590,239,670,239,arrow=True)

text(843,177,"Keep each spot separate",20,weight="bold",anchor="middle")
for i in range(9):
    x=693+i*34
    rect(x,202,26,76,C['orangebg'] if i==0 else C['bluebg'],C['orange'] if i==0 else C['blue'],r=3)
    for j in range(5):
        line(x+6,214+j*11,x+20,214+j*11,color=C['orange'] if i==0 else C['blue'],sw=2)
text(1010,249,"…",26,C['muted'])
text(843,324,"50 spots → 50 tokens",17,C['muted'],anchor="middle")
line(1060,239,1134,239,arrow=True)

rect(1158,184,576,114,C['grey'],r=9)
text(1182,218,"Sparse gene IDs + expression counts",21,weight="bold")
text(1182,249,"Relative 2D coordinates for every spot",19)
text(1182,279,"All source genes considered; no 6,000-gene cutoff",17,C['muted'])
text(1446,324,"GenePT-mapped genes enter the expression encoder",16,C['muted'],anchor="middle")
line(48,352,1752,352,color=C['line'],sw=1)

# b — encoder. Pooling shown conceptually, exact equivalent implementation in caption.
panel('b','GenePT expression encoder · applied independently to every spot',394)
text(119,446,"Expressed genes",19,weight="bold",anchor="middle")
text(366,446,"Frozen GenePT vectors",19,weight="bold",anchor="middle")
text(644,446,"Expression weighting",19,weight="bold",anchor="middle")
for i,(g,x) in enumerate([('g₁','x₁'),('g₂','x₂'),('gₖ','xₖ')]):
    yy=478+i*51
    rect(65,yy-2,112,35,C['grey'],r=4)
    text(121,yy+23,g+' : '+x,20,anchor='middle')
    line(186,yy+16,252,yy+16,arrow=True)
    vector(264,yy+2,n=12,cell=14,h=28,seed=i+1)
    line(461,yy+16,523,yy+16,arrow=True)
    text(492,yy+5,'× '+x,16,anchor='middle')
    vector(538,yy+2,n=12,cell=14,h=28,seed=i+7)
text(367,662,"eᵍ ∈ ℝ¹⁵³⁶",19,C['teal'],anchor='middle')
text(367,690,"Shared gene table",16,C['muted'],anchor='middle')
text(643,662,"xₛᵍ eᵍ",20,C['teal'],anchor='middle')
path('M738,494 L762,494 L762,596 L738,596',color=C['line'])
line(762,545,804,545,arrow=True)
rect(822,477,206,137,C['tealbg'],C['teal'])
text(925,511,"Mean pooling",21,weight='bold',anchor='middle')
text(925,549,"hₛ = Σ xₛᵍeᵍ / kₛ",23,anchor='middle')
text(925,586,"1,536 dimensions",17,C['muted'],anchor='middle')
text(925,650,"kₛ = expressed, mapped",16,C['muted'],anchor='middle')
text(925,674,"unique gene count",16,C['muted'],anchor='middle')
line(1029,545,1071,545,arrow=True)
rect(1089,488,189,116,C['bluebg'],C['blue'])
text(1183,524,"Linear",22,weight='bold',anchor='middle')
text(1183,556,"1,536 → 256",22,anchor='middle')
text(1183,587,"Trainable W + b",17,C['muted'],anchor='middle')
line(1279,545,1390,545,arrow=True)
circle(1411,545,20,'white',C['ink'])
text(1411,553,'+',28,anchor='middle')
line(1432,545,1493,545,arrow=True)
rect(1509,483,223,123,C['orangebg'],C['orange'])
text(1620,519,"Spatial spot token",20,weight='bold',anchor='middle')
text(1620,554,"tₛ = zₛ + pₛ",25,anchor='middle')
text(1620,587,"256 dimensions",18,C['muted'],anchor='middle')

rect(1089,650,246,61,C['bluebg'],C['blue'],r=6)
text(1212,677,"Relative position (Δx, Δy)",17,anchor='middle')
text(1212,700,"Trainable linear: 2 → 256",17,anchor='middle')
path('M1335,681 L1411,681 L1411,566',arrow=True)
text(1425,643,"pₛ",18)
text(73,742,"No neighbouring-spot pooling",17,C['muted'])
text(575,742,"No PCA",17,C['muted'])
text(831,742,"Computed online; no full-slide 1,536-dimensional spot cache",17,C['muted'])
line(48,769,1752,769,color=C['line'],sw=1)

# c — transformer and readout
panel('c','Spatial context modelling and centre-spot prediction',810)
text(161,856,"Token sequence",20,weight='bold',anchor='middle')
for i in range(7):
    rect(68+i*23,880,17,97,C['orangebg'] if i==0 else C['bluebg'],C['orange'] if i==0 else C['blue'],r=2)
text(238,937,'…',23,C['muted'])
text(161,1013,"50 × 256",21,anchor='middle')
text(161,1040,"Centre token first",16,C['muted'],anchor='middle')
line(277,932,320,932,arrow=True)

rect(357,855,335,231,'white',C['line'])
rect(347,865,335,231,'white',C['line'])
rect(337,875,335,231,C['bluebg'],C['blue'])
text(504,908,"Transformer block × 32",22,weight='bold',anchor='middle')
rect(354,929,300,49,'white',C['line'],r=4)
text(504,960,"LayerNorm → SDPA → +",19,anchor='middle')
line(504,979,504,996,arrow=True)
rect(354,1006,300,49,'white',C['line'],r=4)
text(504,1037,"LayerNorm → FFN → +",19,anchor='middle')
text(504,1086,"4 heads · FFN 256 → 512 → 256",16,C['muted'],anchor='middle')
line(693,932,742,932,arrow=True)

rect(758,880,229,105,C['orangebg'],C['orange'])
text(872,912,"Final LayerNorm",20,weight='bold',anchor='middle')
text(872,943,"Read centre token",19,anchor='middle')
text(872,972,"256 dimensions",17,C['muted'],anchor='middle')
text(872,1024,"No CLS token",16,C['muted'],anchor='middle')
text(872,1049,"No sequence pooling",16,C['muted'],anchor='middle')
line(988,932,1033,932,arrow=True)

rect(1050,880,235,105,C['bluebg'],C['blue'])
text(1167,913,"Shared MLP",21,weight='bold',anchor='middle')
text(1167,945,"256 → 4,096 → 1,024",18,anchor='middle')
text(1167,975,"GELU + dropout",17,C['muted'],anchor='middle')
line(1286,932,1315,932)
path('M1315,932 L1315,892 L1351,892',arrow=True)
path('M1315,932 L1315,1029 L1351,1029',arrow=True)

rect(1368,849,364,96,C['tealbg'],C['teal'])
text(1390,880,"Direction head",21,weight='bold')
text(1390,909,"1,024 → 16 angular classes",17)
text(1390,933,"Foreground-masked cross-entropy",14,C['muted'])
cx,cy=1695,877
for i in range(16):
    a1=2*math.pi*i/16; a2=2*math.pi*(i+1)/16
    p1=(cx+22*math.cos(a1),cy+22*math.sin(a1)); p2=(cx+22*math.cos(a2),cy+22*math.sin(a2))
    color=C['teal'] if i==13 else '#CEE1DD'
    s.append(f'<path d="M{cx},{cy} L{p1[0]},{p1[1]} A22,22 0 0,1 {p2[0]},{p2[1]} Z" fill="{color}" stroke="white" stroke-width=".8"/>')
rect(1368,984,364,96,C['orangebg'],C['orange'])
text(1390,1015,"Foreground head",21,weight='bold')
text(1390,1044,"1,024 → 1 foreground logit",17)
text(1390,1068,"Binary cross-entropy",14,C['muted'])
circle(1690,1011,9,C['orange'])
circle(1712,1011,9,'white',C['orange'])
text(1550,1110,"One prediction pair per centre spot",16,C['muted'],anchor='middle')

line(48,1137,1752,1137,color=C['line'],sw=1)
text(48,1177,"GenePT: frozen  ·  Projections, Transformer and heads: trainable",16,C['muted'])
text(1736,1177,"PyTorch  ·  BF16 mixed precision  ·  SDPA  ·  Muon (hidden matrices) + AdamW (other parameters)",16,C['muted'],anchor='end')
s.append('</g></svg>')
(OUT/'architecture.svg').write_text('\n'.join(s), encoding='utf-8')
print(OUT/'architecture.svg')
