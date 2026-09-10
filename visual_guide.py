"""Reader-facing metric purpose for every production chart family."""
GUIDES = [
 ('xg_flow','Each step adds the xG of a shot; flat sections contain no new shot xG. Compare the timing and quality of chances with the score.'),
 ('match_statistic','One page of counted actions for both sides, plus the pressing rate. Every figure is a count from the event stream over the whole match: bars are scaled to the higher of the two, so a long bar is the leader in that row and not a share of anything.'),
 ('shot_map','Location shows where attempts were taken; symbol size encodes xG. Use this to separate shot volume from the quality and location of chances.'),
 ('pass_network','Nodes are touch-based average player locations within the named half; links represent inferred completed passes. Link weight indicates volume, not off-ball positioning.'),
 ('xt_map','Positive xT is the increase in local zone threat from successful passes or explicit carries. Both teams share one colour scale; it is not a probability of scoring.'),
 ('pass_map','Arrows trace recorded pass origins and destinations. Compare route, length and outcome to understand how the ball was moved.'),
 ('goalkeeper','Recorded goalkeeper actions and shot placement describe the attempts faced. Placement alone cannot measure shot-stopping quality or goals prevented.'),
 ('zone14','Zone 14 is the central space immediately outside the penalty area. These actions locate access to a potentially useful shooting and passing area.'),
 ('xt_per_minute','Positive movement xT is grouped by event minute. Peaks locate valuable ball progression, including possessions that did not produce a shot.'),
 ('progressive','Progressive passes move the ball materially closer to goal under the local distance rule. Read origin and destination as well as the total.'),
 ('cross','Cross arrows show delivery location and outcome. A completed cross is a delivery event, not automatically a chance or an assist.'),
 ('defens','Recorded tackles, interceptions, blocks and recoveries locate defensive actions. Event frequency does not measure the positioning of players away from the ball.'),
 ('dominat','Zone shares compare each team’s recorded involvement in the same area. Territorial activity can be high without creating high-quality shots.'),
 ('attacking_zones','Each side’s final-third entries are split across the three lateral corridors, read where the entry landed rather than where it began. A share describes the route taken, not its quality: the corridor that carried the most entries need not be the one that produced the chances.'),
 ('box_entries','Successful movements from outside into the penalty area measure access. Repeated entries in one possession count separately; they are not distinct attacks.'),
 ('high_regain','A high regain is an inferred possession recovery in the advanced zone. Review the next actions to see whether that recovery created box access or a shot.'),
 ('pass_target','Targets are inferred from the next compatible controlled touch after a successful pass. Unmatched passes do not establish a known receiver.'),
 ('ppda','PPDA divides opponent passes by defensive actions in the defined pressing zone. Lower values mean fewer opponent passes per action; it is not a direct measure of pressing success.'),
 ('transition','Transitions are inferred rapid attacks following possession changes. Compare their frequency, duration and shot output using the stated time window.'),
 ('game_state','The score at the time of each event defines the state. Compare output with the minutes spent level, leading or trailing, including added time.'),
 ('player_sequence','xGChain credits every participant in a non-penalty shot possession; xGBuildup excludes shooters and key-pass providers. Teammate credits overlap and cannot be added.'),
 ('momentum','Threat added by successful moves, totalled over a rolling eight minutes and weighted towards the middle of that window. Pressure that never produced a shot is counted; the spells below are cut where the share of threat turned, and each is priced in the shot xG it produced.'),
 ('set_piece','Restart categories separate corners, free kicks and penalties. Compare the number of deliveries or possessions with the xG of the resulting shots.'),
 ('ball_losses','The maps locate open-play losses and immediate opponent output within 12 seconds. Marker outcomes describe the following possession, not off-ball defensive shape.'),
 ('shape','Player touch positions are averaged within time windows. Changes can reflect personnel, possession and score state; these are not tracking-based formations.'),
 ('win_probability','This score/xG heuristic illustrates changing result estimates. It is uncalibrated; the terminal result is known once a completed match is observed.'),
 ('playing_through','Event-based receptions and subsequent progression identify candidate routes through the opponent. Confirm pressure, orientation and positioning in video.'),
 ('action_value','Local action values rank the recorded contributions under the named model. They are model estimates, not an all-round player rating.'),
 ('pitch_control','Time-weighted touch-location windows estimate influence on a 105 × 68 m grid. Substitutions and cards split windows; contested space remains explicit.'),
 ('sequence_types','Possessions are grouped by the recorded sequence rule. Their shots and xG describe output; category names do not prove the coach’s intended approach.'),
 ('goal_origins','Each goal is linked to its recorded buildup sequence and elapsed event time. Use the chain to locate the entry and final action before the finish.'),
 ('press_triggers','Pressing-trigger counts are event-based review cues. They identify sequences to inspect and cannot confirm coordinated pressing without video.'),
]


def guide(path):
    from pathlib import Path
    stem=Path(path).stem.lower()
    return next((text for token,text in GUIDES if token in stem),
                'Read the stated event, time window and denominator before comparing the teams. The figure describes recorded actions in this match.')
