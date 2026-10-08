#!/usr/bin/env python3
"""Reletter the hard-wall and alternative support diagrams without redrawing data.

Requires Matplotlib, NumPy, Pillow, pypdf, and Poppler. No simulations.
The hard-wall support module retains the original drawing with shared typography.
The alternative preserves the original PDF's non-text vector operations.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DictionaryObject, NameObject
import hard_wall_schematic_support as original
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/wall_schematics'
HARD = 'Figure_02_hard_wall'
ALT = 'Figure_02_alt_soft_and_hard_walls'
SOURCES = {
    HARD: ROOT/'data/original_assets/hard_wall_schematic_original.pdf',
    ALT: ROOT/'data/original_assets/ow_overlap_truncation_schematic.pdf',
}
EXPECTED = {
    HARD: '3c765543151639b1d93b04f4c646876ef78188c65aa3eb696a5a20ab1db7b979',
    ALT: 'fbe7ddebeac00bb3d4e64d4053b5a17c025f67906b8185af5744c5e55b2ce99e',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    manuscript_style({'text.color': 'black', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': None})


def rasterize(pdf, png):
    subprocess.run(['pdftoppm','-png','-r','300','-singlefile',str(pdf),str(png.with_suffix(''))], check=True)


def text_boxes(pdf):
    markup = subprocess.check_output(['pdftotext','-bbox',str(pdf),'-'])
    words = ET.fromstring(markup).findall('.//{*}word')
    return [[float(w.attrib[k]) for k in ('xMin','yMin','xMax','yMax')] for w in words]


def build_hard():
    original._configure_style = style
    original.FIGURE_DIR = ROOT
    original.PDF_PATH = ROOT/f'{HARD}.pdf'
    original.PNG_PATH = ROOT/f'{HARD}.png'
    native_panel = original._base_panel
    def relettered_panel(axis):
        native_panel(axis)
        for text in axis.texts:
            text.set_text(text.get_text().replace(r'\boldsymbol{\alpha}', r'\alpha'))
            text.set_color('black')
            text.set_fontweight('normal')
    original._base_panel = relettered_panel
    original.build()


def build_alternative():
    reader = PdfReader(SOURCES[ALT])
    page = reader.pages[0]
    stream = ContentStream(page.get_contents(), reader)
    kept, inside_text, removed_blocks = [], False, 0
    for operands, operator in stream.operations:
        if operator == b'BT':
            assert not inside_text
            inside_text = True
            removed_blocks += 1
        elif operator == b'ET':
            assert inside_text
            inside_text = False
        elif not inside_text:
            kept.append((operands, operator))
    assert not inside_text and removed_blocks == 6
    stream.operations = kept
    page[NameObject('/Contents')] = stream
    # Text was the only font user; remove its unused font resource before overlay.
    page['/Resources'][NameObject('/Font')] = DictionaryObject()
    style()
    figure = plt.figure(figsize=(3.375,1.62), facecolor='none')
    for x, label in ((.035,'(a)'),(.535,'(b)')):
        figure.text(x,.98,label,fontsize=9,fontweight='normal',ha='left',va='top')
    for x, caption in ((64.830339/243,'overlapping support\n(soft wall)'),
                       (193.751839/243,'truncated support\n(hard wall)')):
        figure.text(x,.165,caption,fontsize=8,ha='center',va='top',linespacing=1.05)
    prepare_figure(figure, ALT)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text in figure.texts:
        b = text.get_window_extent(renderer)
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= figure.bbox.width and b.y1 <= figure.bbox.height
    overlay = BytesIO()
    record_typography(figure, ALT)
    figure.savefig(overlay,format='pdf',transparent=True)
    plt.close(figure)
    overlay.seek(0)
    page.merge_page(PdfReader(overlay).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with (ROOT/f'{ALT}.pdf').open('wb') as output:
        writer.write(output)
    rasterize(ROOT/f'{ALT}.pdf', ROOT/f'{ALT}.png')
    return {'original_text_blocks_replaced': removed_blocks,
            'original_nontext_vector_operations_retained': len(kept)}


def compare_diagram(stem, scratch):
    oldpng = scratch/f'{stem}.png'
    rasterize(SOURCES[stem], oldpng)
    before = np.asarray(Image.open(oldpng).convert('RGB'))
    after = np.asarray(Image.open(ROOT/f'{stem}.png').convert('RGB'))
    assert before.shape == after.shape
    text_mask = np.zeros(before.shape[:2], dtype=bool)
    # Mask old and new glyph bounds only, including a two-point antialias margin.
    for pdf in (SOURCES[stem], ROOT/f'{stem}.pdf'):
        for x0,y0,x1,y1 in text_boxes(pdf):
            x0,y0 = np.floor((np.array([x0,y0])-2)*300/72).astype(int)
            x1,y1 = np.ceil((np.array([x1,y1])+2)*300/72).astype(int)
            text_mask[max(0,y0):min(before.shape[0],y1+1),max(0,x0):min(before.shape[1],x1+1)] = True
    difference = np.max(np.abs(before.astype(int)-after.astype(int)),axis=2)
    changed = int(np.count_nonzero(difference[~text_mask]))
    assert changed == 0, (stem, 'Diagram pixels changed outside glyph bounds', changed)
    with Image.open(ROOT/f'{stem}.png') as image:
        assert all(abs(d-300)<.05 for d in image.info['dpi'])
    return {'png_shape':list(before.shape),'nontext_pixels_changed':changed,
            'unmasked_diagram_pixels_checked':int((~text_mask).sum()),
            'source_sha256':sha(SOURCES[stem]),
            'outputs':{f'{stem}.{ext}':sha(ROOT/f'{stem}.{ext}') for ext in ('pdf','png')}}


def main():
    for stem in SOURCES:
        assert sha(SOURCES[stem]) == EXPECTED[stem]
    DATA.mkdir(parents=True,exist_ok=True)
    build_hard()
    vector_check = build_alternative()
    with tempfile.TemporaryDirectory() as work:
        records = {stem:compare_diagram(stem,Path(work)) for stem in SOURCES}
    receipt = {'status':'passed','figures':records,'alternative_vector_check':vector_check,
               'typography':'Computer Modern text/math through LaTeX; normal 9 pt panel letters; schematic labels 10–11 pt at manuscript inclusion width',
               'diagram_geometry_colors_shapes_unchanged':True,
               'native_support_sha256':sha(Path(original.__file__)),
               'renderer_sha256':sha(__file__),
               'requirements':['Matplotlib','NumPy','Pillow','pypdf','Poppler','LaTeX','dvipng','cm-super']}
    (DATA/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
