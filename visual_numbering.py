"""Number published figures by display order, keeping semantic names intact."""
import re
import json
from pathlib import Path


def number_visuals(paths, out, contracts):
    out = Path(out).resolve()
    mapping = {}
    pending = []
    index = 0
    for source in paths:
        source = Path(source).resolve()
        match = re.match(r'^\d+[a-z]?_(.+)$', source.name)
        if source.parent != out or not match:
            continue
        index += 1
        target = out / f'{index:02d}_{match.group(1)}'
        mapping[source] = target
        temporary = source.with_suffix('.renumbering')
        source.rename(temporary)
        pending.append((temporary, target))
    for temporary, target in pending:
        temporary.rename(target)
    names = {source.name: target.name for source, target in mapping.items()}
    updated = {names.get(key, key): value for key, value in contracts.items()}
    contracts.clear()
    contracts.update(updated)
    contract_file = out / 'analysis_tables' / 'chart_contracts.json'
    if contract_file.exists():
        contract_file.write_text(json.dumps(contracts, indent=2, ensure_ascii=False), encoding='utf-8')
    return [mapping.get(Path(path).resolve(), Path(path).resolve()) for path in paths]
