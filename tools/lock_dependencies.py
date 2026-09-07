"""Write an exact dependency lock from a successful pip resolution report."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

parser=argparse.ArgumentParser()
parser.add_argument('report',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--prefer-installed',action='store_true')
args=parser.parse_args()
report=json.loads(args.report.read_text(encoding='utf-8'))
rows=[]
for item in report['install']:
    name=item['metadata']['name'];chosen=item['metadata']['version']
    if args.prefer_installed:
        try:chosen=version(name)
        except PackageNotFoundError:pass
    rows.append(f'{name}=={chosen}')
args.output.write_text('# Python 3.12 dependency lock; generated from pip resolution.\n'+'\n'.join(sorted(rows,key=str.lower))+'\n',encoding='utf-8')
print(f'{len(rows)} exact versions written to {args.output}')
