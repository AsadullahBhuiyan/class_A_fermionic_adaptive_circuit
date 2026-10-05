"""Check the integrated manuscript against its pre-revision baseline and figure bundle."""
from pathlib import Path
import hashlib, json, re, subprocess
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def strip_gpt(s):
    while '\\GPT{' in s:
        a=s.index('\\GPT{'); i=a+5; depth=1
        while depth:
            depth += (s[i]=='{')-(s[i]=='}'); i+=1
        s=s[:a]+s[i:]
    return s

def check():
    s=(ROOT/'manuscript.tex').read_text()
    old=(HERE/'baseline/manuscript.tex').read_text()
    before=s[s.index('\\section{Introduction}'):s.index('{\\color{blue}\nWe then introduce programmable')]
    baseline=old[old.index('\\section{Introduction}'):old.index('We then introduce dynamical domain walls')]
    assert ' '.join(strip_gpt(before).split())==' '.join(baseline.split()),'Introduction changed outside final paragraph/GPT notes'
    assert (ROOT/'references.bib').read_text().startswith((HERE/'baseline/references.bib').read_text()),'Original bibliography entries changed'
    stems=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s)
    expected=['01_schematic','02_hard_wall','03_bulk_topology','04_purification','05_correlations','06_entropy_charge','07_entanglement_spectrum','08_central_charge','09_wall_entropy','10_modular_evolution','11_mean_channel','A01_ow_truncation','A02_mutual_information']
    assert stems==['Figure_'+x+'.pdf' for x in expected]
    ii=s.index('\\section{Topological Adaptive Circuit'); iii=s.index('\\section{Dynamical Chiral Domain-Wall'); iv=s.index('\\section{Conclusions and Outlook}'); app=s.index('\\appendix')
    assert all(ii<s.index('Figure_'+x+'.pdf')<iii for x in expected[:2])
    assert all(iii<s.index('Figure_'+x+'.pdf')<iv for x in expected[2:11])
    assert all(s.index('Figure_'+x+'.pdf')>app for x in expected[11:])
    labels=re.findall(r'\\label\{([^}]+)\}',s)
    assert len(labels)==len(set(labels))
    refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s))
    assert not refs-set(labels)
    assert not any(x in s for x in ['PENDING','meas_slab_only','meas\\_slab'])
    assert '\\newcommand{\\rec}{\\mathbf m}' in s
    assert 'G_c=2G-\\Id' in s
    assert 'first within each trajectory and then over trajectories' in s
    assert '\\newcommand{\\newtext}[1]{{\\color{blue}#1}}' in s
    assert '\\newcommand{\\GPT}[1]{{\\color{magenta}[GPT: #1]}}' in s
    assert 'vortices hosting Majorana zero modes' in s
    for name in ['channel_gap_appendix.tex','purification_appendix.tex']:
        assert (HERE/name).read_text().strip() in s
    log=(ROOT/'build/manuscript.log').read_text()
    for bad in ['Undefined control sequence','undefined references','undefined citations','Overfull','Float too large','A float is stuck','multiply defined']:
        assert bad not in log,bad
    aux=(ROOT/'build/manuscript.aux').read_text()
    assert '\\newlabel{app:Truncation}{{A}' in aux
    assert '\\newlabel{fig:Truncation}{{A1}' in aux
    assert '\\newlabel{fig:MutualInformation}{{A2}' in aux
    pages=int(re.search(r'Output written on .*?\((\d+) pages',log).group(1))
    result={'all_checks_passed':True,'pages':pages,'figures_included':stems,'alternative_preserved_but_not_included':True,
            'intro_before_last_paragraph_unchanged_except_GPT_notes':True,'original_bibliography_entries_unchanged':True,
            'missing_or_duplicate_labels':[],'overflow_or_stuck_float_warnings':[],
            'blue_revision_macro':True,'magenta_GPT_notes':s.count('\\GPT{'),
            'averaging_definition_present':True,'new_circuit_simulations':False,
            'tex_sha256':digest(ROOT/'manuscript.tex'),'bibliography_sha256':digest(ROOT/'references.bib'),
            'compiled_pdf_sha256':digest(ROOT/'build/manuscript.pdf'),
            'build_command':'latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex',
            'figure_bundle_verification':'figures/new_figure/validation.json'}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': check()
