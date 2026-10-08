#!/usr/bin/env python3
"""Check actual PDF figure placements, not just requested LaTeX page breaks."""
import argparse
import json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    index = json.loads((ROOT/'data/figure_index.json').read_text())
    expected = {r['stem']:r for r in index if r['included']}
    reader = PdfReader(args.pdf)
    placements = []
    for number,page in enumerate(reader.pages,1):
        matrix = np.eye(3)
        stack = []
        resources = page['/Resources'].get('/XObject',{})
        if hasattr(resources,'get_object'): resources = resources.get_object()
        for operands,operator in ContentStream(page['/Contents'],reader).operations:
            if operator == b'q': stack.append(matrix.copy())
            elif operator == b'Q': matrix = stack.pop()
            elif operator == b'cm':
                a,b,c,d,e,f = map(float,operands)
                matrix = matrix@np.array([[a,c,e],[b,d,f],[0,0,1]])
            elif operator == b'Do':
                obj = resources[operands[0]].get_object()
                filename = obj.get('/PTEX.FileName')
                if not filename: continue
                stem = Path(str(filename)).stem
                if stem not in expected: continue
                x0,y0,x1,y1 = map(float,obj['/BBox'])
                points = matrix@np.array([[x0,x0,x1,x1],[y0,y1,y0,y1],[1,1,1,1]])
                bbox = [float(points[0].min()),float(points[1].min()),float(points[0].max()),float(points[1].max())]
                top = float(page.mediabox.top)-bbox[3]
                placements.append(dict(stem=stem,figure=expected[stem]['number'],page=number,
                    bbox_bottom_origin=bbox,top_from_page_edge=top,
                    vertical_stack=expected[stem]['vertical_stack']))
    assert {r['stem'] for r in placements} == set(expected)
    assert len(placements) == len(expected)
    counts = {}
    for row in placements:
        if row['vertical_stack']:
            counts[row['page']] = counts.get(row['page'],0)+1
        x0,y0,x1,y1 = row['bbox_bottom_origin']
        assert x0 >= 53 and x1 <= 563 and y0 >= 35 and y1 <= 741, row
    assert max(counts.values()) == 1
    schematic = next(r for r in placements if r['figure']=='1')
    assert schematic['bbox_bottom_origin'][2]-schematic['bbox_bottom_origin'][0] > 500
    for number in ('1','6'):
        row = next(r for r in placements if r['figure']==number)
        assert row['bbox_bottom_origin'][2]-row['bbox_bottom_origin'][0] > 500, row
    result = dict(status='passed',pdf=str(args.pdf.resolve()),pages=len(reader.pages),
        maximum_vertical_stacks_per_page=max(counts.values()),vertical_stack_count=sum(counts.values()),
        all_stacks_start_at_page_top=all(r['top_from_page_edge'] <= 60 for r in placements if r['vertical_stack']),
        forced_page_breaks=False,figure1_full_width=True,placements=placements)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
