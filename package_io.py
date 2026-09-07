"""Recoverable package replacement and reproducible provenance."""
from pathlib import Path
from functools import wraps
import hashlib
import inspect
import json
import shutil
import tempfile
import uuid
import time
from datetime import datetime, timezone
from metric_registry import METRIC_VERSION, definitions

def _copy_unlocked(src, dst):
    try:
        shutil.copy2(src, dst)
    except PermissionError:
        # A viewer may hold one stale publication file; the newly generated
        # staging tree remains valid and the next run can replace it.
        pass


def write_manifest(out, info):
    out=Path(out)
    sources={}
    for name in ['events.csv','players.csv','xg.csv','team_advanced_metrics.csv','player_sequence_metrics.csv','match_info.json']:
        p=out/name
        if p.exists():sources[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    chart_file=out/'analysis_tables'/'chart_contracts.json'
    manifest={'schema_version':1,'metric_version':METRIC_VERSION,'generated_utc':datetime.now(timezone.utc).isoformat(),
              'fixture':info,'inputs':sources,'definitions':definitions(),
              'models':{'xG':'input snapshot; calibration provenance must be supplied by the collector',
                        'post_shot':'placement heuristic, uncalibrated','xT':'input grid values, local approximation',
                        'influence':'duration-weighted 5-minute touch windows split at substitutions/cards; 105 x 68 m; uncalibrated',
                        'win_probability':'uncalibrated score/xG heuristic; completed terminal result is known'},
              'charts':json.loads(chart_file.read_text(encoding='utf-8')) if chart_file.exists() else {},
              'outputs':sorted(str(p.relative_to(out)).replace('\\','/') for p in out.rglob('*') if p.is_file() and p.suffix in ['.png','.pdf','.docx','.md','.csv'])}
    (out/'package_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def transactional_package(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        bound=inspect.signature(function).bind(*args,**kwargs);bound.apply_defaults()
        original=Path(bound.arguments['output_dir']).resolve()
        if original==original.parent or original.name in ['', '.', '..']:
            raise ValueError('A dedicated match output directory is required')
        original.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.'+original.name+'.stage-',dir=original.parent)).resolve()
        if stage.parent != original.parent:
            raise ValueError('Staging must stay beside the match output')
        try:
            # Build into a clean staging tree. Copying the previous package
            # first can fail when Word/ImageMagick/Explorer has one of its
            # files open, and every published artifact is regenerated from
            # the supplied snapshot anyway.
            bound.arguments['output_dir']=stage
            bound.arguments['events'].to_csv(stage/'events.csv',index=False,encoding='utf-8-sig')
            result=function(*bound.args,**bound.kwargs)
            if not result.get('article') or not Path(result['pdf']).is_file():
                raise RuntimeError('Publication incomplete: original package preserved; inspect '+str(stage))
            write_manifest(stage,bound.arguments['match_info'])
            backup=original.with_name('.'+original.name+'.previous-'+uuid.uuid4().hex[:8])
            published_in_place=False
            # Windows can briefly lock the destination directory (usually an
            # image viewer or the antivirus indexer). Retry the backup move;
            # if it remains locked, publish the validated staging tree in
            # place instead of failing the complete render.
            backup_error=None
            if original.exists():
                for attempt in range(8):
                    try:
                        original.rename(backup)
                        backup_error=None
                        break
                    except PermissionError as error:
                        backup_error=error
                        time.sleep(0.5*(attempt+1))
            if backup_error is not None:
                shutil.copytree(stage,original,dirs_exist_ok=True,copy_function=_copy_unlocked)
                shutil.rmtree(stage,ignore_errors=True)
                backup=None
                published_in_place=True
            try:
                # Windows can briefly hold a generated PDF/PNG while the light
                # child process and antivirus indexer release it. Retry the
                # atomic rename, then fall back to a same-volume copy so a
                # completed package is not reported as failed.
                if not published_in_place:
                    last_error=None
                    for attempt in range(5):
                        try:
                            stage.rename(original)
                            last_error=None
                            break
                        except PermissionError as error:
                            last_error=error
                            time.sleep(0.5*(attempt+1))
                    if last_error is not None:
                        shutil.copytree(stage,original,dirs_exist_ok=False)
                        shutil.rmtree(stage)
            except Exception:
                if backup is not None and backup.exists() and not original.exists():backup.rename(original)
                raise
            def relocate(item):
                if isinstance(item,Path):
                    try:return original/item.relative_to(stage)
                    except ValueError:return item
                if isinstance(item,list):return [relocate(v) for v in item]
                if isinstance(item,dict):return {k:relocate(v) for k,v in item.items()}
                return item
            result=relocate(result)
            # A successful replacement is the published state. Do not leave a
            # second match folder beside it; callers can opt into backups at
            # the filesystem level if they need historical snapshots.
            if backup is not None and backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            # Restore module globals for callers that inspect the renderer after use.
            function.__globals__['OUT']=original
            return result
        except Exception:
            # Keep failed staging for diagnosis; never delete the last good output.
            print('Incomplete staging retained at '+str(stage))
            raise
    return wrapped
