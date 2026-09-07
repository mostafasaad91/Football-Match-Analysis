"""Regenerate poster visuals for every locally saved match package."""
from pathlib import Path
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parent/'output'

def main():
    packages=[]
    for info in ROOT.rglob('match_info.json'):
        folder=info.parent
        if (folder/'events.csv').exists() and (folder/'players.csv').exists() and folder.name!='light':
            packages.append(folder)
    jobs=[]
    for folder in sorted(set(packages)):
        for theme, destination in [('dark',folder),('light',folder/'light')]:
            destination.mkdir(exist_ok=True)
            jobs.append((folder,theme,destination))
    def render(job):
        folder,theme,destination=job
        command=[sys.executable,'render_posters.py',str(folder),'--output',str(destination),'--theme',theme]
        result=subprocess.run(command,cwd=Path(__file__).resolve().parent,capture_output=True,text=True)
        return job,result
    ok=0; failed=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(render,j) for j in jobs]):
            job,result=future.result()
            if result.returncode==0: ok+=1
            else: failed.append((str(job[0]),job[1],result.stderr[-300:]))
    print(f'packages={len(packages)} renders={ok} failures={len(failed)}')
    for item in failed: print(item)

if __name__=='__main__': main()
