"""Dump what BeautifulSoup, markdownify and the Chandra/Surya output parsers make of a corpus of HTML.

    python tools/reference/dump_html.py --chandra /path/to/chandra --surya /path/to/surya \
        --out tests/data/reference/html/html.json

The corpus is model-shaped output (layout divs, tables, math, lists, figures) plus malformed
markup and a seeded fuzz of fragments, because what matters for parity is what html.parser and
bs4 make of output that is not quite HTML. Each case records ``soup`` (``str(BeautifulSoup(x,
"html.parser"))``), Chandra's ``parse_html`` / ``parse_markdown`` (default and inclusive flags) /
``parse_chunks`` on a 1000x1400 page with its image names, and Surya's parsers where they apply.
Requires markdownify==1.1.0 (Chandra's pin).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

CASES = [
    "",
    "plain text only",
    "<p>Hello <b>world</b></p>",
    '<div data-label="Text" data-bbox="10 20 500 80">Some plain text without tags</div>',
    '<div data-label="Text" data-bbox="10 20 500 80"><p>Para one</p><p>Para two with *stars* and _under_ and $dollars$</p></div>',
    '<div data-label="Section-Header" data-bbox="10 10 990 40"><h1>Title</h1></div>'
    '<div data-label="Page-Header" data-bbox="0 0 1000 10"><p>Header</p></div>'
    '<div data-label="Page-Footer" data-bbox="0 990 1000 1000"><p>7</p></div>',
    '<div data-label="Image" data-bbox="100 200 600 700"><img alt="A chart of sales"/>Bars rising</div>',
    '<div data-label="Figure" data-bbox="100 200 600 700">No image tag here</div>',
    '<div data-label="Text" data-bbox="1 2 3 4"><p>text <img alt="x"> with img</p></div>',
    '<div data-label="Blank-Page" data-bbox="0 0 1000 1000"></div><div data-label="Text" data-bbox="5 5 50 50">After blank</div>',
    '<div data-label="Table" data-bbox="50 100 950 400"><table border="1"><thead><tr><th>A</th><th colspan="2">B</th></tr></thead>'
    '<tbody><tr><td>1</td><td>2</td><td>3</td></tr><tr><td rowspan="2">x &amp; y</td><td>&lt;5</td><td>a>b</td></tr></tbody></table></div>',
    '<div data-label="Equation-Block" data-bbox="10 10 20 20"><math display="block">\\frac{a}{b} < c</math></div>',
    '<div data-label="Text" data-bbox="0 0 1 1"><p>Inline <math>x^2 + y_1</math> math and <math display="inline">\\alpha</math>.</p></div>',
    '<div data-label="List-Group" data-bbox="0 0 1 1"><ul><li>One</li><li>Two<ul><li>Nested a</li><li>Nested b</li></ul></li><li>Three</li></ul></div>',
    '<div data-label="List-Group" data-bbox="0 0 1 1"><ol start="3"><li>c</li><li>d</li></ol><p>after list</p></div>',
    '<div data-label="Code-Block" data-bbox="0 0 1 1"><pre><code>def f(x):\n    return x * 2\n</code></pre></div>',
    '<div data-label="Text" data-bbox="0 0 1 1"><p>Link <a href="http://x.org">http://x.org</a> and <a href="http://y.org" title="T">named [y]</a></p></div>',
    '<div data-label="Form" data-bbox="0 0 1 1"><p><input type="checkbox" checked> Accept <input type="radio" value="a"> A</p></div>',
    '<div data-label="Text" data-bbox="0 0 1 1"><p>H<sub>2</sub>O and E=mc<sup>2</sup>, <i>italic</i>, <u>under</u>, <del>gone</del></p></div>',
    '<div data-label="Chemical-Block" data-bbox="0 0 1 1"><chem>CC(=O)O</chem></div>',
    '<div data-label="Text" data-bbox="bad box">bad bbox</div><div data-label="Text">no bbox</div><div>no label</div>',
    '<div data-label="Text" data-bbox="10 20 30">three numbers</div><div data-label="Text" data-bbox="-5 -5 2000 2000">clamped</div>',
    '<div data-label="Text" data-bbox="1 2 3 4"><div data-bbox="5 6 7 8" data-label="Inner"><p>nested bbox</p></div></div>',
    'top level text <div data-label="Text" data-bbox="1 1 2 2">div</div> trailing',
    "<p>unclosed paragraph<p>second<b>bold never closed",
    "<p>a<br>b</br>c<br/>d</p>",
    "<div>&amp; &lt; &gt; &quot; &#65; &#x42; &bogus; &nbsp; &copy &amp</div>",
    '<p title="a\'b">q</p><p title=\'say "hi"\'>r</p><p title="both \' and &quot;">s</p>',
    '<div class="a  b   c" id=x data-x=1>attrs</div>',
    "<math display=\"block\">a<b</math>",
    "<!-- comment --><p>after comment</p><!DOCTYPE html><?pi data?>",
    "<script>if (a < b && c > d) {}</script><style>p > a { color: red }</style>",
    "<table><tr><td>no tbody</td><td>x</td></tr><tr><td>y</td><td>z</td></tr></table>",
    "<h2>Heading  with   spaces</h2>\n\n<h7>bad heading</h7><h3>Third</h3>",
    "<blockquote><p>Quoted text</p><p>Second</p></blockquote>",
    "<p>   leading and trailing   </p>   <p>\n\nnewlines\n\n</p>",
    "<div>\n  <p>x</p>\n  </div>\n",
    "<p>Price: $5 and 10% off * deals _now_</p>",
    "<ul><li>a</li>\n<li>b\n</li></ul>",
    "<p>Tabs\tand\t\tspaces</p>",
    "<span>inline</span><span>span</span>",
    "<p>x</p><hr><p>y</p>",
    '<img src="a.png" alt="alt text">',
    "<p>unicode: café — “quotes” 中文 😀</p>",
    "<pre>  keep   spacing\n\tand tabs</pre>",
    "<code>a_b*c</code> outside_code",
    "</p>stray close<div></span>mismatched</div>",
    "<p>end with ampersand &",
    "<p>end with lt <",
    "<div data-label=\"Text\" data-bbox=\"0 0 10 10\"><p>a</p></div><div data-label=\"Table\" data-bbox=\"0 0 10 10\"><table><tr><th>h</th></tr><tr><td>c</td></tr></table></div>",
    "<b> spaced bold </b> and <i></i>empty",
    "<p>a<br/>b<br/>c</p><td>loose cell</td>",
    "<DIV DATA-LABEL=\"Text\" DATA-BBOX=\"1 2 3 4\"><P>UPPER</P></DIV>",
    "<p>&#0; &#128; &#x110000; &#55296; &#99999999999;</p>",
    "<a href=\"x\">[link] (text)</a>",
    "<dl><dt>Term</dt><dd>Definition line</dd></dl>",
]

FRAGMENTS = [
    "<p>", "</p>", "<div>", "</div>", "<b>", "</b>", "<i>", "</i>", "<br>", "<br/>", "<img alt='x'>", "<table>", "</table>",
    "<tr>", "</tr>", "<td>", "</td>", "<th colspan='2'>", "<ul>", "</ul>", "<li>", "</li>", "<math>", "</math>",
    '<math display="block">', "text", " ", "\n", "\t", "&amp;", "&lt;", "&", "<", ">", "*", "_", "$", "x^2", "a < b",
    '<div data-label="Text" data-bbox="1 2 3 4">', '<div data-label="Image" data-bbox="10 10 90 90">', "<!-- c -->",
    "<sup>", "</sup>", "<sub>", "</sub>", "<pre>", "</pre>", "<code>", "</code>", "<a href='u'>", "</a>", "<h2>", "</h2>",
]


def fuzz(n: int, seed: int = 1234) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(3, 25))) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chandra", required=True)
    ap.add_argument("--surya", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="text files of raw model output to add")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.chandra)
    from bs4 import BeautifulSoup
    from PIL import Image

    from chandra.output import Markdownify, extract_images, parse_chunks, parse_html, parse_markdown

    converter = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )

    cases = CASES + fuzz(300) + [Path(p).read_text() for p in args.extra]
    image = Image.new("RGB", (1000, 1400), (200, 100, 50))
    results = []
    for html in cases:
        entry = {"html": html, "soup": str(BeautifulSoup(html, "html.parser"))}
        entry["chandra_html"] = parse_html(html)
        entry["chandra_html_all"] = parse_html(html, include_headers_footers=True, include_images=False)
        entry["chandra_markdown"] = parse_markdown(html)
        entry["chandra_markdown_all"] = parse_markdown(html, include_headers_footers=True, include_images=False)
        entry["markdownify"] = converter.convert(html)
        chunks = parse_chunks(html, image)
        entry["chandra_chunks"] = chunks
        entry["chandra_images"] = {k: list(v.size) for k, v in extract_images(html, chunks, image).items()}
        results.append(entry)

    if args.surya:
        sys.path.insert(0, args.surya)
        from surya.inference import parsers

        for entry in results:
            html = entry["html"]
            entry["surya_clean_block_html"] = parsers.clean_block_html(html)
            try:
                entry["surya_full_page"] = [
                    [b["label"], b["bbox"], b["html"]] for b in parsers.parse_full_page_html(html)
                ]
            except Exception as e:  # noqa: BLE001 - recorded, not fatal
                entry["surya_full_page_error"] = type(e).__name__

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=0))
    print(f"{len(results)} cases -> {out}")


if __name__ == "__main__":
    main()
