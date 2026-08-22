"""Where a match belongs: which competition, which season, which round.

The output folder was flat. Fifty fixtures from six competitions sat beside
each other in one directory, sorted by nothing, and finding last weekend's
Premier League matches meant reading folder names. This works out the shelf a
fixture goes on so the pipeline can file it: ``England_Premier_League/2026-2027
/Matchweek_03/Arsenal_vs_Coventry_3-0``.

The competition and season come out of the URL, which carries them in a fixed
shape:

    /matches/1997563/live/portugal-liga-portugal-2026-2027-casa-pia-ac-benfica
                          └ region ┘└ competition ┘└ season ┘└── teams ──┘

The teams are the awkward part — the slug runs them together with no separator
that survives a hyphenated club name. They are not needed: the fixture already
knows who played, so the parse only has to find where the season ends and stop.

The round is not in the URL and not in the match feed, so it is not invented.
A fixture whose round cannot be established is filed under the season directly,
which is honest and still sorted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# england-championship-2026-2027-wolves-blackburn
#  region ┘ competition ┘ season ┘
_URL_SLUG = re.compile(
    r"/matches/\d+/[a-z]+/"
    r"(?P<slug>[a-z0-9-]+?)-"
    r"(?P<season>\d{4}-\d{4}|\d{4})-"
)

# The leading token is the region, and the rest is the competition — except
# where the region is two words. Listing those is shorter and safer than
# guessing from the shape of the slug.
_TWO_WORD_REGIONS = (
    "united-states", "south-africa", "south-korea", "saudi-arabia",
    "new-zealand", "czech-republic", "north-macedonia", "bosnia-herzegovina",
    "costa-rica", "el-salvador", "hong-kong", "ivory-coast", "north-ireland",
    "northern-ireland", "republic-of-ireland", "san-marino", "faroe-islands",
    "united-arab-emirates", "trinidad-and-tobago",
)


@dataclass(frozen=True)
class Fixture:
    """The shelf a match belongs on."""

    region: str            # "England"
    competition: str       # "Premier League"
    season: str            # "2026-2027"
    round_name: str        # "Matchweek 03", or "" when it is not known

    @property
    def competition_folder(self) -> str:
        """``England_Premier_League``, or ``Unsorted`` when unknown."""
        parts = [p for p in (self.region, self.competition) if p]
        return _folder("_".join(parts)) if parts else "Unsorted"

    @property
    def parts(self) -> tuple[str, ...]:
        """The directories a match sits under, outermost first."""
        found = [self.competition_folder]
        if self.season:
            found.append(_folder(self.season))
        if self.round_name:
            found.append(_folder(self.round_name))
        return tuple(found)


def _folder(text: str) -> str:
    """A name a filesystem will take, on every platform."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return cleaned.strip("._") or "Unsorted"


def _titled(slug: str) -> str:
    """"premier-league" -> "Premier League", keeping known casings."""
    special = {"uefa": "UEFA", "fa": "FA", "efl": "EFL", "mls": "MLS",
               "afc": "AFC", "caf": "CAF", "concacaf": "CONCACAF",
               "laliga": "LaLiga", "psg": "PSG", "usa": "USA"}
    words = []
    for word in slug.split("-"):
        words.append(special.get(word, word.capitalize()))
    return " ".join(words)


def from_url(url: str | None) -> Fixture:
    """Read the competition and season out of a WhoScored match URL."""
    found = _URL_SLUG.search(str(url or "").lower())
    if not found:
        return Fixture("", "", "", "")

    slug = found.group("slug")
    season = found.group("season")

    region_slug = slug.split("-")[0]
    for candidate in _TWO_WORD_REGIONS:
        if slug.startswith(candidate + "-"):
            region_slug = candidate
            break

    competition_slug = slug[len(region_slug):].strip("-")
    return Fixture(
        region=_titled(region_slug),
        competition=_titled(competition_slug),
        season=season,
        round_name="",
    )


def describe(fixture: Fixture) -> str:
    """One line naming the competition, for the report's own header."""
    parts = [p for p in (fixture.region, fixture.competition) if p]
    line = " ".join(parts)
    if fixture.season:
        line = f"{line} {fixture.season}".strip()
    if fixture.round_name:
        line = f"{line} · {fixture.round_name}".strip()
    return line


def round_from_date(played_on: str | None) -> str:
    """The week a fixture was played in, as a round when none was given.

    WhoScored does not publish the matchweek in the URL or the match feed, so
    there is nothing to read. What the feed does carry is the date, and a
    league round is a weekend: grouping by the Monday that starts the week puts
    the round together without inventing a number for it.

    Named for what it is. Calling it "Matchweek 3" when nothing in the data
    says three would be a guess wearing a fact's clothes.
    """
    from datetime import date, datetime, timedelta

    text = str(played_on or "").strip()
    if not text:
        return ""
    for shape in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                  "%d/%m/%Y", "%d-%m-%Y"):
        try:
            when = datetime.strptime(text[:len(shape) + 4], shape).date()
            break
        except ValueError:
            continue
    else:
        return ""
    monday = when - timedelta(days=when.weekday())
    return f"Week_of_{monday.isoformat()}"


def shelf(url: str | None, round_name: str = "",
          played_on: str | None = None) -> tuple[str, ...]:
    """The directories one fixture belongs under, outermost first.

    ``round_name`` wins when the caller knows the matchweek. Without it the
    date groups the round, which needs nothing from the user and is right often
    enough to be useful.
    """
    fixture = from_url(url)
    name = str(round_name or "").strip() or round_from_date(played_on)
    if name:
        fixture = Fixture(fixture.region, fixture.competition,
                          fixture.season, name)
    return fixture.parts
