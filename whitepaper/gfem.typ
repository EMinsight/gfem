// gfem document style for Typst: the docs site's look (docs/_static/gfem.css) set for print.
//
// A panel-grey ground, dark-grey ink, hairline divisions and one signal red.
// Words in Schibsted Grotesk, numbers and code in JetBrains Mono. Flat: depth is
// the panel grey inside a hairline, never a shadow.
//
// Build with `just whitepaper` (data from tools/whitepaper_data.py, then typst).

// ---------------------------------------------------------------- tokens

#let c = (
  ground: rgb("#f2f2f2"),
  panel: rgb("#fafafa"),
  ink: rgb("#262626"),
  ink-2: rgb("#3d3d3d"),
  muted: rgb("#6b6b6b"),
  hair: rgb("#dcdcdc"),
  mesh: rgb("#c4c4c4"),
  chip: rgb("#e6e6e4"),
  accent: rgb("#b5262f"),
  accent-soft: rgb("#ecd5d6"),
  hll: rgb(181, 38, 47, 7%),
  ok: rgb("#1a7354"),
  led-ok: rgb("#39c296"),
  bad: rgb("#a64600"),
  warn: rgb("#8a5a00"),
)

#let sans = ("Schibsted Grotesk",)
#let mono = ("JetBrains Mono", "DejaVu Sans Mono")

// The left gutter carries section numbers and margin notes, like the printed
// scale beside a dial. `gutter` is how far into the margin they reach.
#let margin-left = 50mm
#let gutter = 36mm

// ---------------------------------------------------------------- type voices

// Label: mono uppercase, letter-spaced. Never for text that may hold "gfem" or a value.
#let legend(body, fill: c.muted, size: 6.6pt) = text(
  font: mono, size: size, tracking: 0.08em, fill: fill, weight: 400,
  if type(body) == str and body.clusters().len() == 1 { body } else { upper(body) },
)

// Data: mono sentence case, for values, tolerances, page numbers.
#let data(body, fill: c.ink-2, size: 7.4pt) = text(font: mono, size: size, fill: fill, body)

// Readout value: the large mono number beside a label.
#let value(body, size: 15pt) = text(font: mono, size: size, tracking: -0.02em, fill: c.ink, body)

// ---------------------------------------------------------------- marks

// LED: a round indicator, the only status and location mark.
#let led(fill: c.accent, size: 4.6pt) = box(
  baseline: -0.5pt, circle(radius: size / 2, fill: fill, stroke: none),
)

// The Shape-Beside-Hue Rule: every status hue sits beside its shape and a word.
#let mark(status, word: auto) = {
  let (shape, text-fill, default) = if status == "verified" {
    (box(baseline: -0.3pt, circle(radius: 2.9pt, fill: c.led-ok)), c.ok, "FD-verified")
  } else if status == "zero" {
    (box(baseline: -2.6pt, line(length: 7pt, stroke: 1pt + c.muted)), c.muted, "exact zero")
  } else if status == "raises" {
    (
      box(baseline: 0.4pt, width: 8.5pt, height: 8.5pt, {
        place(circle(radius: 4.25pt, stroke: 1pt + c.bad))
        place(line(start: (1.2pt, 7.3pt), end: (7.3pt, 1.2pt), stroke: 1pt + c.bad))
      }),
      c.bad,
      "raises",
    )
  } else {
    (box(baseline: -0.3pt, circle(radius: 2.9pt, stroke: 1pt + c.ink)), c.ink, status)
  }
  let w = if word == auto { default } else { word }
  box(shape)
  if w != none { h(4pt); data(w, fill: text-fill) }
}

// ---------------------------------------------------------------- blocks

// Readout strip: mono label + value pairs under an ink top rule, spread apart.
#let readout(..pairs, rule: c.ink) = block(width: 100%, above: 10pt, below: 6pt, {
  line(length: 100%, stroke: 0.6pt + rule)
  v(-2pt)
  pairs.pos().map(((k, v)) => box[#legend(k) #h(3pt) #data(v, fill: c.ink)]).join(h(1fr))
})

// Note (admonition): square panel in a hairline, mono title led by an LED in its hue.
#let note(kind: "note", title: auto, body) = {
  let hue = (
    note: c.muted, tip: c.led-ok, important: c.ink, warning: c.warn, danger: c.bad,
  ).at(kind)
  let t = if title == auto { kind } else { title }
  block(
    width: 100%, fill: c.panel, stroke: 0.5pt + c.hair, inset: (x: 12pt, y: 10pt),
    above: 12pt, below: 12pt, breakable: false,
    {
      led(fill: hue, size: 5.4pt); h(5pt); legend(t, fill: c.ink)
      v(4pt)
      set text(size: 8.6pt, fill: c.ink-2)
      body
    },
  )
}

// Panel: the face plate; something is raised by sitting on it inside a hairline.
#let panel(body, inset: 10pt) = block(
  width: 100%, fill: c.panel, stroke: 0.5pt + c.hair, inset: inset, body,
)

// Wide: run a block into the left gutter (figures and the gradient matrix only).
#let wide(body) = move(dx: -gutter, block(width: 100% + gutter, body))

// Margin note: small muted text in the left gutter, top-aligned with the next line.
#let aside(body) = place(
  left, dx: -gutter, block(width: gutter - 6mm, {
    set par(justify: false, leading: 0.5em)
    set text(size: 7pt, fill: c.muted)
    body
  }),
)

// Code panel with an optional decisive line: a 1.5pt red left edge over the red wash.
#let code(hl: (), body) = {
  show raw.line: it => {
    if it.number in hl {
      box(
        width: 100%, fill: c.hll, outset: (left: 12pt, right: 12pt, y: 2.2pt),
        stroke: (left: 1.5pt + c.accent), it,
      )
    } else { it }
  }
  body
}

// Key: a labelled control, 4pt corners, panel fill, hairline edge.
#let key(body, fill: c.panel, stroke: c.mesh, ink: c.ink) = box(
  fill: fill, stroke: 0.5pt + stroke, radius: 2.4pt, inset: (x: 5pt, y: 3.4pt),
  baseline: 3.4pt, text(font: mono, size: 7pt, fill: ink, body),
)

// ---------------------------------------------------------------- figures

// |S21|^2 per iteration: an ink line, hair ticks where the projection sharpens.
#let history(t, beta, height: 58pt) = layout(size => {
  let W = size.width
  let n = t.len()
  let x(i) = W * i / (n - 1)
  let y(v) = height * (1 - v)
  box(width: W, height: height + 12pt, {
    // scale lines at 0 and 1
    place(line(start: (0pt, y(1)), end: (W, y(1)), stroke: 0.4pt + c.hair))
    place(line(start: (0pt, y(0)), end: (W, y(0)), stroke: 0.6pt + c.ink))
    // projection sharpens: beta doubles
    for i in range(1, n) {
      if beta.at(i) != beta.at(i - 1) {
        place(line(start: (x(i), y(0)), end: (x(i), y(1)), stroke: 0.4pt + c.mesh))
        place(dx: x(i) + 2pt, dy: y(0) - 8pt, data("β " + str(int(beta.at(i))), fill: c.muted, size: 5.8pt))
      }
    }
    place(curve(
      stroke: 1pt + c.ink,
      curve.move((x(0), y(t.at(0)))),
      ..range(1, n).map(i => curve.line((x(i), y(t.at(i))))),
    ))
    place(dx: x(n - 1) - 2.2pt, dy: y(t.at(n - 1)) - 2.2pt, circle(radius: 2.2pt, fill: c.ink))
    place(dx: 0pt, dy: height + 3pt, data("iteration 0", fill: c.muted, size: 5.8pt))
    place(right, dy: height + 3pt, data(str(n - 1), fill: c.muted, size: 5.8pt))
  })
})

// A design snapshot: the PNG on the panel, the design square dashed over it.
// `square` is the design square's share of the box side.
#let snapshot(path, square: 0.6) = layout(size => {
  let w = size.width
  box(width: w, height: w, fill: c.panel, stroke: 0.5pt + c.hair, {
    place(image(path, width: w, height: w))
    place(dx: w * (1 - square) / 2, dy: w * (1 - square) / 2, rect(
      width: w * square, height: w * square,
      stroke: (paint: c.mesh, thickness: 0.5pt, dash: "dashed"),
    ))
  })
})

// ---------------------------------------------------------------- document

#let paper(
  title: none,
  short-title: none,
  kicker: "Whitepaper",
  lead: none,
  version: none,
  date: none,
  meta: (),
  hero: none,
  authors: (),
  print: false,
  body,
) = {
  let ground = if print { white } else { c.ground }
  set document(title: title)
  set text(font: sans, size: 9.4pt, fill: c.ink-2, lang: "en", hyphenate: false)
  set par(leading: 0.68em, spacing: 1.05em, justify: false)
  set page(paper: "a4", fill: ground)

  // ------------------------------------------------------------ cover
  page(margin: (x: 20mm, top: 18mm, bottom: 18mm), {
    grid(
      columns: (1fr, auto),
      align: (left + horizon, right + horizon),
      image("data/gfem-lockup-light.svg", height: 20pt),
      [#legend(kicker, fill: c.ink) #h(6pt) #data(version, fill: c.muted)],
    )
    v(4pt)
    line(length: 100%, stroke: 0.5pt + c.hair)
    v(1fr)
    block(width: 72%, text(
      size: 40pt, weight: 500, tracking: -0.032em, fill: c.ink, font: sans,
      par(leading: 0.16em, title),
    ))
    v(12pt)
    block(width: 64%, text(size: 13pt, fill: c.ink-2, par(leading: 0.5em, lead)))
    if authors.len() > 0 {
      v(8pt)
      text(size: 10pt, fill: c.ink, authors.join(", "))
    }
    v(1fr)
    if hero != none { hero }
    v(14pt)
    readout(..meta)
  })

  // ------------------------------------------------------------ body pages
  set page(
    margin: (left: margin-left, right: 20mm, top: 26mm, bottom: 22mm),
    header-ascent: 34%,
    footer-descent: 30%,
    header: context {
      // The section in force at the top of the page, or the one that opens it.
      let here-page = here().page()
      let before = query(heading.where(level: 1).before(here()))
      let opening = query(heading.where(level: 1)).filter(h => h.location().page() == here-page)
      let at-top = opening.filter(h => h.location().position().y < 45mm)
      let current = if at-top.len() > 0 { at-top.first() } else if before.len() > 0 { before.last() } else if opening.len() > 0 { opening.first() } else { none }
      move(dx: -gutter, block(width: 100% + gutter, {
        grid(
          columns: (1fr, auto),
          align: (left + bottom, right + bottom),
          [#box(baseline: 1.6pt, image("data/gfem-mark-light.svg", height: 7.5pt)) #h(5pt) #data(short-title, fill: c.muted)],
          if current != none {
            let n = counter(heading).at(current.location()).first()
            [#led() #h(4pt) #data(if n < 10 { "0" + str(n) } else { str(n) }, fill: c.muted) #h(4pt) #data(current.body, fill: c.ink)]
          },
        )
        v(-4pt)
        line(length: 100%, stroke: 0.5pt + c.hair)
      }))
    },
    footer: context move(dx: -gutter, block(width: 100% + gutter, {
      line(length: 100%, stroke: 0.5pt + c.hair)
      v(-4pt)
      grid(
        columns: (1fr, auto),
        data([gfem #version · #date], fill: c.muted),
        data(
          {
            let p = counter(page).get().first()
            let total = counter(page).final().first()
            (if p < 10 { "0" } else { "" }) + str(p) + " / " + (if total < 10 { "0" } else { "" }) + str(total)
          },
          fill: c.ink,
        ),
      )
    })),
  )
  counter(page).update(2)

  // ------------------------------------------------------------ headings
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => {
    let n = counter(heading).at(it.location()).first()
    block(width: 100%, above: 22pt, below: 11pt, breakable: false, sticky: true, {
      move(dx: -gutter, line(length: 100% + gutter, stroke: 0.5pt + c.hair))
      v(7pt)
      place(dx: -gutter, dy: 3.5pt, legend(if n < 10 { "0" + str(n) } else { str(n) }, fill: c.accent, size: 7.4pt))
      text(size: 16pt, weight: 500, tracking: -0.012em, fill: c.ink, it.body)
    })
  }
  show heading.where(level: 2): it => block(above: 15pt, below: 7pt, sticky: true, {
    let nums = counter(heading).at(it.location())
    place(dx: -gutter, dy: 1.6pt, data(nums.map(str).join("."), fill: c.muted, size: 6.8pt))
    text(size: 10.6pt, weight: 600, fill: c.ink, it.body)
  })
  show heading.where(level: 3): it => block(above: 12pt, below: 6pt, sticky: true,
    text(size: 9.4pt, weight: 600, fill: c.ink, it.body))

  // ------------------------------------------------------------ inline
  show strong: set text(weight: 600, fill: c.ink)
  show emph: set text(style: "italic")
  // Ink Link Rule: links are ink on a mesh underline; red is for hover only.
  show link: it => underline(stroke: 0.5pt + c.mesh, offset: 2pt, evade: true, text(fill: c.ink, it))
  show raw: set text(font: mono)
  show raw.where(block: false): it => box(
    fill: c.chip, radius: 1.6pt, inset: (x: 1.8pt), outset: (y: 1.8pt),
    text(size: 0.9em, fill: c.ink, it),
  )
  set raw(theme: "gfem.tmTheme")
  show raw.where(block: true): it => block(
    width: 100%, fill: c.panel, stroke: 0.5pt + c.hair, inset: (x: 12pt, y: 10pt),
    above: 10pt, below: 12pt,
    { set par(leading: 0.62em); text(size: 7.6pt, it) },
  )

  // ------------------------------------------------------------ math
  set math.equation(numbering: n => data("(" + str(n) + ")", fill: c.muted))
  show math.equation.where(block: true): set block(above: 12pt, below: 12pt)

  // ------------------------------------------------------------ lists
  set list(marker: box(baseline: -2.2pt, line(length: 5pt, stroke: 0.6pt + c.muted)), indent: 2pt, body-indent: 7pt)
  set enum(numbering: n => data(str(n) + ".", fill: c.muted), indent: 0pt, body-indent: 7pt)

  // ------------------------------------------------------------ tables
  // Horizontal rules only: ink top rule over label heads, hairline rows.
  set table(
    stroke: (x, y) => (
      top: if y == 0 { 0.6pt + c.ink } else if y == 1 { 0.5pt + c.ink } else { 0.5pt + c.hair },
    ),
    inset: (x: 5pt, y: 5pt),
    align: left + horizon,
  )
  show table.cell.where(y: 0): it => legend(it, fill: c.muted, size: 6.2pt)
  show table: set text(size: 8pt)
  show table: it => block(width: 100%, {
    it
    v(-6pt)
    line(length: 100%, stroke: 0.5pt + c.hair)
  })

  // ------------------------------------------------------------ figures
  set figure(gap: 7pt)
  show figure: set block(above: 14pt, below: 14pt)
  show figure.where(kind: table): set figure.caption(position: top)
  show figure.caption: it => {
    set align(left)
    set par(leading: 0.55em)
    let num = context it.counter.display(it.numbering)
    let kind = if it.kind == table { "Table" } else { "Fig." }
    block(width: 100%, [#legend(kind + " " + num, fill: c.ink) #h(5pt) #text(size: 7.8pt, fill: c.muted, it.body)])
  }

  body
}
