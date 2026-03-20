PART 1 — FORMAL SPECIFICATION (SATTVA-Deal v1)
1. Objective

Given:

Historical labeled successful deal streams 
𝐷
=
{
𝑑
1
,
𝑑
2
,
.
.
.
,
𝑑
𝑛
}
D={d
1
	​

,d
2
	​

,...,d
n
	​

}

Each deal 
𝑑
𝑖
d
i
	​

 is a typed temporal event graph

Learn:

A bank of structural invariants (primitives) 
𝑃
P

Such that:

For a new incoming stream 
𝑆
S, we compute:

𝑆
𝑐
𝑜
𝑟
𝑒
(
𝑆
)
=
∑
𝑝
∈
𝑃
match
(
𝑆
,
𝑝
)
Score(S)=
p∈P
∑
	​

match(S,p)

Escalate to LLM when Score exceeds threshold.

2. Event Representation

Each deal stream becomes a graph:

Nodes:

Events 
𝑒
e

Entities (company, person, product, funding, etc.)

Edges:

Temporal order

Causal relation

Role/attribute relation

Each event has vector embedding:

𝑣
𝑒
∈
𝑅
𝑘
v
e
	​

∈R
k

But geometry is not in embedding magnitude.
Geometry is in relational constraints.

3. Geometric Construction per Deal

For each deal:

Embed events into vector space.

Construct relationship vectors:

𝑟
𝑖
𝑗
=
𝑣
𝑗
−
𝑣
𝑖
r
ij
	​

=v
j
	​

−v
i
	​


Construct weighted adjacency matrix:

𝐴
𝑖
𝑗
=
𝑓
(
𝑟
𝑖
𝑗
,
Δ
𝑡
𝑖
𝑗
,
𝑡
𝑦
𝑝
𝑒
𝑖
,
𝑡
𝑦
𝑝
𝑒
𝑗
)
A
ij
	​

=f(r
ij
	​

,Δt
ij
	​

,type
i
	​

,type
j
	​

)

Extract small relational subgraphs:

2-node chains

3-node chains

3-node cycles

4-node motifs

Each motif becomes:

𝑀
=
(
𝑠
𝑡
𝑟
𝑢
𝑐
𝑡
𝑢
𝑟
𝑒
,
𝑐
𝑜
𝑛
𝑠
𝑡
𝑟
𝑎
𝑖
𝑛
𝑡
𝑠
𝑖
𝑔
𝑛
𝑎
𝑡
𝑢
𝑟
𝑒
)
M=(structure,constraintsignature)

Where constraint signature includes:

Relative magnitude ratios

Time-normalized intervals

Type patterns

4. Primitive Definition

A primitive 
𝑝
p satisfies:

Appears in ≥ m independent deals.

Increases predictive lift over baseline.

Survives scale/time normalization.

Has low entropy across occurrences.

Formally:

𝐿
𝑖
𝑓
𝑡
(
𝑝
)
=
𝑃
(
𝑠
𝑢
𝑐
𝑐
𝑒
𝑠
𝑠
∣
𝑝
)
−
𝑃
(
𝑠
𝑢
𝑐
𝑐
𝑒
𝑠
𝑠
)
Lift(p)=P(success∣p)−P(success)
𝑆
𝑡
𝑎
𝑏
𝑖
𝑙
𝑖
𝑡
𝑦
(
𝑝
)
=
𝑉
𝑎
𝑟
(
𝑠
𝑖
𝑔
𝑛
𝑎
𝑡
𝑢
𝑟
𝑒
(
𝑝
)
)
−
1
Stability(p)=Var(signature(p))
−1

Keep primitives maximizing:

𝑆
𝑐
𝑜
𝑟
𝑒
(
𝑝
)
=
𝐿
𝑖
𝑓
𝑡
(
𝑝
)
⋅
𝑆
𝑡
𝑎
𝑏
𝑖
𝑙
𝑖
𝑡
𝑦
(
𝑝
)
−
𝜆
𝐶
𝑜
𝑚
𝑝
𝑙
𝑒
𝑥
𝑖
𝑡
𝑦
(
𝑝
)
Score(p)=Lift(p)⋅Stability(p)−λComplexity(p)
5. Matching Function

For incoming stream S:

For each primitive p:

𝑚
𝑎
𝑡
𝑐
ℎ
(
𝑆
,
𝑝
)
=
exp
⁡
(
−
𝛼
⋅
𝑠
𝑡
𝑟
𝑢
𝑐
𝑡
𝑢
𝑟
𝑎
𝑙
_
𝑑
𝑖
𝑠
𝑡
𝑎
𝑛
𝑐
𝑒
(
𝑆
,
𝑝
)
)
match(S,p)=exp(−α⋅structural_distance(S,p))

Total score:

𝑆
𝑐
𝑜
𝑟
𝑒
(
𝑆
)
=
∑
𝑝
𝑤
𝑝
⋅
𝑚
𝑎
𝑡
𝑐
ℎ
(
𝑆
,
𝑝
)
Score(S)=
p
∑
	​

w
p
	​

⋅match(S,p)

Escalation threshold is learned via validation set.

6. Design Principles

No blind unsupervised abstraction.

Supervised structural compression.

Invariance under time scaling and magnitude scaling.

Explicit relational geometry.

Interpretability: every escalation tied to activated motifs.

This is your sellable thesis.
