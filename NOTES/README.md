# Notes library

This folder is the single documentation front door. It currently indexes 208 canonical
documents. Standalone root papers and notes now live physically under this folder;
documentation belonging to current project directories remains at its project path and
is exposed here through relative links.

| Section | Contents |
|---|---|
| `00_CURRENT/` | Numerical campaign working note, production README, audit, layout, and current manuscript |
| `10_MANUSCRIPTS/` | PRXQ/PRL drafts, introduction, results summary, and repository synthesis |
| `20_THEORY_AND_METHODS/` | Internal reference sheets, derivations, standalone theory PDFs, and method-specific docs |
| `30_PROJECT_DOCUMENTATION/` | Documentation attached to runnable packages and campaigns |
| `40_EXTERNAL_PAPERS/` | Physically housed external/source papers and temporary reference papers |
| `90_LEGACY/` | Useful older notes and working records |
| `ALL_BY_SOURCE/` | Linked mirror of documents that remain in separate project/administrative folders |
| `RECENCY_INDEX.md` | Every indexed document sorted by modification time |

The `standalone/`, `sources/`, and `root_imports/` directories contain physical files
moved from the repository root. Other curated entries are relative symbolic links;
editing through them edits the canonical project document. Generated LaTeX debris,
result figures, caches, and `*Notes.bib` files are intentionally excluded.

Rebuild/add newly created documents with:

```bash
python scripts/build_notes_view.py
```

The builder refuses missing curated targets and refuses to replace a nonmatching path.
