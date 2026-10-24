from newsroom.outlets import outlet_of, same_outlet
from newsroom.registry import load_sources


def test_all_city_channels_are_one_outlet():
    for name in (
        "City of Oulu",
        "Mun Oulu (City of Oulu magazine)",
        "BusinessOulu",
        "Oulun kaupunki · ilmoitukset",
    ):
        assert outlet_of(name) == "City of Oulu"


def test_public_transport_is_its_own_outlet():
    assert outlet_of("Oulu Region Public Transport (OSL)") == "OSL"
    assert not same_outlet("OSL", "City of Oulu")


def test_every_migri_feed_is_one_outlet():
    names = {s.name for s in load_sources() if "Migri" in s.name or "Immigration" in s.name}
    assert {outlet_of(n) for n in names} == {"Migri"}


def test_configured_official_feeds_are_enabled_and_have_urls():
    official = [s for s in load_sources() if s.official and s.enabled]
    ids = {s.id for s in official}
    assert {"oulu-city", "oulu-notices", "migri-press", "migri-bulletins"} <= ids
    for source in official:
        assert source.feed_url and source.feed_url.startswith("https://")
        assert source.access == "full"
