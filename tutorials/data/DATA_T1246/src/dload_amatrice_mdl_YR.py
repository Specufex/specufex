import obspy
import obspy.core as oc
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain,
    MassDownloader,
    RectangularDomain,
    Restrictions,
)

# Circular Domain, Miao Zhang had used 50km
domain = CircularDomain(latitude=42.75, longitude=13.25, minradius=0.0, maxradius=0.7)

# wfBaseDir='/Users/mameier/data/project/amatrice19/wf'    # laptop
wfBaseDir = "/data/hy73/felixw/Amatrice/waveforms1"

restrictions = Restrictions(
    # Get data for a whole year.
    # (YenJoe's picks go from 2016-08-15 - 2017-08-15)
    starttime=obspy.UTCDateTime(2016, 8, 15),
    endtime=obspy.UTCDateTime(2017, 8, 15),
    # Chunk it to have one file per day.
    chunklength_in_sec=86400,
    # Considering the enormous amount of data associated with continuous
    # requests, you might want to limit the data based on SEED identifiers.
    # If the location code is specified, the location priority list is not
    # used; the same is true for the channel argument and priority list.
    network="YR",
    # network="IV", #network=["YR","IV"],
    # network=None, station=None, location=None, channel=None,
    # channel_priorities=["HHZ"],
    channel_priorities=["[HE][HN]?"],
    # channel_priorities=["[HE][HN][ZEN]"],
    # The typical use case for such a data set are noise correlations where
    # gaps are dealt with at a later stage.
    reject_channels_with_gaps=False,
    # Same is true with the minimum length. All data might be useful.
    minimum_length=0.0,
    # Guard against the same station having different names.
    minimum_interstation_distance_in_m=100.0,
)
# Restrict the number of providers if you know which serve the desired
# data. If in doubt just don't specify - then all providers will be
# queried.
mdl = MassDownloader(providers=["IRIS", "INGV"])
mdl.download(
    domain,
    restrictions,
    mseed_storage=wfBaseDir + "/{network}/{station}/"
    "{channel}.{location}.{starttime}.{endtime}.mseed",
    stationxml_storage="stations",
)
