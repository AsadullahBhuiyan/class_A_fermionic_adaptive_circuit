from pathlib import Path
import nbformat,json,hashlib,shutil
from nbclient import NotebookClient
OUT=Path(__file__).resolve().parent
shutil.copytree(OUT.parent/'pure_half_system_energy_size_scan_L099_v1/latex_support',OUT/'latex_support',dirs_exist_ok=True)
nb=nbformat.read(OUT/'analysis_template.ipynb',as_version=4)
NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,OUT/'entanglement_level_statistics.ipynb')
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.is_file() and p.name!='completion_manifest.json'}
(OUT/'completion_manifest.json').write_text(json.dumps({'status':'complete','files':files},indent=2)+'\n')
print('WALL-ONLY LEVEL FIGURE COMPLETE:',OUT)
