// ─── Page & Typography Setup ───────────────────────────────────────────────────
#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  numbering: "1",
  number-align: right,
)

#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.72em, spacing: 1.2em)
#set heading(numbering: none)

#show math.equation: set text(font: "New Computer Modern Math")

#let pd(top, bottom) = $frac(partial #top, partial #bottom)$
#let ddot(s) = $dot(dot(#s))$

// ─── Accent colour ─────────────────────────────────────────────────────────────
#let accent = rgb("#1a5276")   // deep navy-blue

// ─── Section heading style ─────────────────────────────────────────────────────
#show heading.where(level: 1): it => {
  v(1.4em)
  block(
    fill: accent,
    inset: (x: 10pt, y: 6pt),
    radius: 3pt,
    width: 100%,
  )[
    #text(fill: white, weight: "bold", size: 12pt)[#it.body]
  ]
  v(0.6em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(fill: accent, weight: "bold", size: 11pt)[#it.body]
  v(0.4em)
}

// ─── Figure helper: cap height so figures never dominate a page ────────────────
#show figure: it => {
  block(width: 100%)[
    #it.body
    #if it.caption != none {
      v(0.3em)
      align(center)[
        #text(size: 9.5pt, style: "italic")[
          #it.supplement #context it.counter.display(). #it.caption.body
        ]
      ]
    }
  ]
  v(0.5em)
}

// ─── Header (no names – two-column layout) ─────────────────────────────────────
#set page(header: [
  #set text(size: 9pt)
  #grid(
    columns: (1fr, 1fr),
    align: (left, right),
    [Robot and Reinforcement Learning], [*Coursework 2: PPO vs SAC*],
  )
  #line(length: 100%, stroke: 0.4pt + luma(160))
])


= PPO Hyperparameter Tuning

- chose clip coefficient (epsilon) and GAE lambda for Hyperparameter tuning 
  - $epsilon$: clips update step size for policy parameters
  - $lambda$: trades off between bias and variance ; low $lambda$ $=>$ high bias $=>$ like TD , high $lambda$ $=>$ high variance $=>$ like Monte Carlo Method

#figure(
  table(
    columns: (auto, auto, auto, auto,),
    align: (left, right, right, right),
    stroke: 0.5pt,
    table.header(
      [], 
      [*$lambda = 0.90$*], 
      [*$lambda = 0.95$*], 
      [*$lambda = 1.00$*]
    ),
    [*$epsilon = 0.1$*], [1471.80], [1378.69], [1657.81],
    [*$epsilon = 0.2$*],  [*4449.39*], [2706.83], [2397.13],
    [*$epsilon = 0.3$*],   [1131.88], [*4568.06*], [150.90],
  ),
  caption: [Episode return for different values of $epsilon$ and $lambda$],
) <tab1>

- Two sets of $epsilon$ & $lambda$ values have shown good performance with close episodic returns
  - $[epsilon = 0.2, lambda = 0.9]$ : Average Episodic Return = 4449.39
  -  $[epsilon = 0.3, lambda = 0.95]$ : Average Episodic Return = 4568.06