"""
Shared team-name normalisation used across modules.

Extracted to avoid circular imports between ``world_cup_model`` and
``player_features``.
"""

TEAM_ALIASES = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "D.R. Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Türkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "United States": "USA",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
}


def canonical_team(name: str) -> str:
    cleaned = str(name).strip()
    return TEAM_ALIASES.get(cleaned, cleaned)
