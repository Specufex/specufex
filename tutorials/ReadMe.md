# Tutorial for SpecUFEx: The Geysers, California

Written by Theresa Sawi and Nate Groebner

Based on the study **Machine learning reveals cyclic changes in seismic source spectra in Geysers geothermal field** by Holtzman et al., 2018. DOI 10.1126/sciadv.aao2929

_________

**What's happening in run_tutorial.ipynb**



0. Follow instructions for installing SpecUFEx at https://github.com/Specufex/specufex
1. Read in waveforms from local folder
2. Convert waveforms to spectrograms (filtered and median normalized)
3. Run SpecUFEx on spectrograms
4. Do kmeans clustering on SpecUFEx fingerprints
5. Compare clusters to paper figure 3c
