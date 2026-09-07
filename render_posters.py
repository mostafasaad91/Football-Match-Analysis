"""Rebuild only the four posters from a saved match, without network access."""
import argparse
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--theme',choices=['dark','light'],default='dark')
    args=parser.parse_args()
    os.environ['MATCH_ANALYSIS_THEME']=args.theme
    import matplotlib
    matplotlib.use('Agg')
    from render_snapshot import load_snapshot
    from poster_dashboard import build_match_posters
    events,players,xg,teams,player_metrics,info=load_snapshot(args.source)
    paths=build_match_posters(events,xg,teams,player_metrics,players,out_dir=args.output,
        home_id=info['home_id'],away_id=info['away_id'],
        home_name=info['home_name'],away_name=info['away_name'],
        home_color=info['home_color'],away_color=info['away_color'],score=info['score'],
        competition=info.get('competition','MATCH ANALYSIS'),match_date=info.get('date',''),
        allow_download=False)
    for path in paths:print(path)


if __name__=='__main__':main()
