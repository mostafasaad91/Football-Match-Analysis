"""Offline re-render of a saved fixture, without scraping or external writes."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from package_io import transactional_package


@transactional_package
def refresh_publication(events,players,xg,team_metrics,player_metrics,match_info,output_dir,*,source):
    """Reuse saved boards, regenerate the analytical extension and publications."""
    import shutil
    from insight_visuals import build_insight_visuals
    from tactical_pdf_report import build_tactical_pdf
    from match_article import build_match_article
    from visual_redesign_full import configure_match, build_catalog
    out=Path(output_dir)
    for p in Path(source).glob('*.png'):shutil.copy2(p,out/p.name)
    for name,frame in [('players',players),('xg',xg),('team_advanced_metrics',team_metrics),('player_sequence_metrics',player_metrics)]:frame.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    (out/'match_info.json').write_text(json.dumps(match_info,indent=2,ensure_ascii=False),encoding='utf-8')
    configure_match(match_info,out)
    # Rebuild the boards whose calculations or labels changed.
    import visual_redesign_full as visual
    visual.xt_map(events,match_info['home_id'],7);visual.xt_map(events,match_info['away_id'],8)
    visual.control_surface(events);visual.win_probability_curve(events)
    visual.shot_map(events,xg,match_info['home_id'],2);visual.shot_map(events,xg,match_info['away_id'],3)
    visual.turnovers(events,match_info['home_id'],37);visual.turnovers(events,match_info['away_id'],38)
    new,_=build_insight_visuals(events,players,match_info,out)
    paths=sorted(set(out.glob('[0-9]*.png'))|set(new))
    build_catalog([p.resolve() for p in paths])
    pdf=build_tactical_pdf(paths,out/'full_visual_redesign_real_data.pdf',events,xg,team_metrics,player_metrics,match_info)
    article=build_match_article(events,xg,team_metrics,player_metrics,match_info,out,players)
    from poster_dashboard import build_match_posters
    for old in out.glob('match_poster_*.png'):old.unlink()
    posters=build_match_posters(events,xg,team_metrics,player_metrics,players,out_dir=out,
        home_id=match_info['home_id'],away_id=match_info['away_id'],home_name=match_info['home_name'],away_name=match_info['away_name'],
        home_color=visual.HOME,away_color=visual.AWAY,score=match_info['score'],match_date=match_info.get('date',''))
    return {'output_dir':out,'pdf':pdf,'article':article,'visuals':paths,'posters':posters,'catalog':out/'visual_catalog.csv'}


def load_snapshot(source):
    source=Path(source)
    info=json.loads((source/'match_info.json').read_text(encoding='utf-8-sig'))
    names=['events','players','xg','team_advanced_metrics','player_sequence_metrics']
    frames=[pd.read_csv(source/(name+'.csv')) for name in names]
    return (*frames,info)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--theme',choices=['light','dark'],default='dark')
    parser.add_argument('--both-themes',action='store_true')
    parser.add_argument('--publication-only',action='store_true',help='Reuse saved boards in the selected theme; rebuild changed boards and publications')
    args=parser.parse_args()
    os.environ['MATCH_ANALYSIS_THEME']=args.theme
    os.environ['MATCH_ANALYSIS_LIGHT_COPY']='1' if args.both_themes else '0'
    from visual_redesign_full import generate_match_package
    if args.publication_only:
        result=refresh_publication(*load_snapshot(args.source),args.output,source=args.source)
    else:
        result=generate_match_package(*load_snapshot(args.source),args.output)
        if args.both_themes and not result.get('light_output_dir'):
            raise RuntimeError('The requested light package did not finish; inspect the render log.')
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
