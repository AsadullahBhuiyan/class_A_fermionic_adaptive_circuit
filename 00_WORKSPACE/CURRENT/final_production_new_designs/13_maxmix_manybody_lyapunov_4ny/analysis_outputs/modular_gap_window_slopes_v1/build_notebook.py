from pathlib import Path
import nbformat,json,hashlib
from nbclient import NotebookClient
out=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook(metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
for section in (out/'analysis.py').read_text().split('# %%'):
    if not section.strip():continue
    if section.startswith(' markdown'):
        lines=section.splitlines()[1:]
        source='\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line for line in lines)
        nb.cells.append(nbformat.v4.new_markdown_cell(source))
    else:nb.cells.append(nbformat.v4.new_code_cell(section.strip()))
p=out/'modular_gap_window_slopes.ipynb';nbformat.write(nb,p)
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
nbformat.write(nb,p)
files={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in out.iterdir()
 if f.is_file() and f.name not in ['completion_manifest.json','run.log','exit_code.txt']}
(out/'completion_manifest.json').write_text(json.dumps(dict(status='complete',files=files),indent=2)+'\n')
print('COMPLETE',out)
