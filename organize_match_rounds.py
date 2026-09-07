"""Normalize saved match folders to explicit Matchweek names."""
from pathlib import Path
import json
import shutil
from match_fixture import round_from_date

ROOT=Path(__file__).resolve().parent/'output'

def main():
    renamed=0; moved=0
    # Existing generated seasons used bare numeric folders (1, 2, ...).
    for competition in ROOT.iterdir():
        if not competition.is_dir() or competition.name in {'raw_snapshots','poster_redesign','profile_preview'}: continue
        for season in competition.iterdir():
            if not season.is_dir(): continue
            for folder in list(season.iterdir()):
                if folder.is_dir() and folder.name.isdigit():
                    target=season/f'Matchweek_{int(folder.name):02d}'
                    if not target.exists():
                        folder.rename(target); renamed+=1
    # The bundled sample predates fixture metadata. It is a Premier League
    # sample, so place it beside the generated league packages by date week.
    sample=ROOT/'Arsenal_vs_Coventry_3-0_Final'
    info_path=sample/'match_info.json'
    if sample.exists() and info_path.exists():
        info=json.loads(info_path.read_text(encoding='utf-8-sig'))
        week=round_from_date(info.get('date')) or 'Unclassified'
        target=ROOT/'England_Premier_League'/'2026-2027'/week/sample.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.move(str(sample),str(target)); moved+=1
    print(f'renamed={renamed} moved={moved}')

if __name__=='__main__': main()
