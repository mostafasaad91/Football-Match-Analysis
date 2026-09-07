"""Visual comparisons preserve denominators, missing values and shared scales."""
import numpy as np
import matplotlib.pyplot as plt
from poster_dashboard import Poster, _display_number, _pair_scale


def test_percent_tracks_keep_their_own_denominator():
    values,ceiling=_pair_scale('25%','50%')
    assert values==[25,50]
    assert ceiling==100


def test_counts_share_a_scale_and_missing_is_not_zero():
    values,ceiling=_pair_scale('—','8')
    assert np.isnan(values[0]) and values[1]==8 and ceiling==8
    assert _display_number('0')==0
    assert np.isnan(_display_number('Unavailable'))
    assert _pair_scale('0','0')[1]>0


def test_percent_bars_encode_quarter_and_half_not_pair_share(monkeypatch):
    import crests
    monkeypatch.setattr(crests,'place_crest',lambda *a,**kw:False)
    info=dict(home_id=1,away_id=2,home_name='Home',away_name='Away',
              home_color='#EF0107',away_color='#78D2F2',score='0–0')
    board=Poster(info,1,'Comparisons','',[])
    try:
        board.comparisons(0,'Rates',[('Success','25%','50%')])
        bars=board.axes[0].patches
        assert np.isclose(bars[1].get_width(),.48*.25)
        assert np.isclose(bars[3].get_width(),.48*.50)
        assert all(bar.get_x()==.40 for bar in bars)
    finally:
        plt.close(board.fig)
