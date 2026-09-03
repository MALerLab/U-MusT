# Linearized MusicXML codes from [OLiMPiC 1.0 dataset](https://github.com/ufal/olimpic-icdar24)

This repository contains the source code for the article:

> **Practical End-to-End Optical Music Recognition for Pianoform Music**<br>
> https://link.springer.com/chapter/10.1007/978-3-031-70552-6_4

## MusicXML Linearization

Linearized MusicXML is a sequential format convertible to and from MusicXML (with minimal losses), that can be used to train img2seq ML models. The conversions are implemented in the `app.linearization` module by classes `Linearizer` and `Delinearizer`. These classes are built to convert a single system (or a part when ignoring system breaks). But you can use the CLI wrapper instead:

```bash
# MusicXML -> LMX (accepts both XML and MXL)
python3 -m app.linearization linearize example.musicxml # produces example.lmx
python3 -m app.linearization linearize example.mxl # produces example.lmx
cat input.musicxml | python3 -m app.linearization linearize - # prints to stdout (only uncompressed XML input)

# LMX -> MusicXML (only uncompressed XML output available)
python3 -m app.linearization delinearize input.lmx # produces example.musicxml
cat input.lmx | python3 -m app.linearization delinearize - # prints to stdout
```

The `app.linearization.vocabulary` module defines all the LMX tokens.

To read more about the linearization process, see the [`docs/linearized-musicxml.md`](docs/linearized-musicxml.md) documentation file.

## Licenses

Datasets are available under CC BY-SA license.

- OLiMPiC synthetic
- OLiMPiC scanned
- GrandStaff-LMX (only the added `.lmx` and `.musicxml` files)

The trained Zeus model is available under CC BY-SA license (available for download in the releases page).

Source code in this repository is available under the MIT license.


## Acknowledgement

If you publish material based on the LMX linearizer or the LMX data, we request you to include a reference to paper \[1\]. If you use the OLiMPiC dataset, you should also include a reference to the OpenScore Lieder corpus \[2\]. If you use the GrandStaff-LMX dataset, you should include a reference to the GradStaff dataset \[3\].


## References

\[1\] Jiří Mayer, Milan Straka, Jan Hajič jr., Pavel Pecina. Practical End-to-End Optical Music Recognition for Pianoform Music. *18th International Conference on Document Analysis and Recognition, ICDAR 2024.* Athens, Greece, August 30 - September 4, pp. 55-73, 2024.<br>
**DOI:** https://doi.org/10.1007/978-3-031-70552-6_4<br>
**GitHub:** https://github.com/ufal/olimpic-icdar24

\[2\] Mark Gotham, Robert Haigh, Petr Jonas. The OpenScore Lieder Corpus. *Music Encoding Conference.* Online, July 19-22, pp. 131-136, 2022.<br>
**DOI:** https://doi.org/10.17613/1my2-dm23<br>
**GitHub:** https://github.com/OpenScore/Lieder

\[3\] Antonio Ríos-Villa, David Rizo, José M. Iñesta, Jorge Calvo-Zaragoza. End-to-end optical music recognition for pianoform sheet music. *International Journal on Document Analysis and Recognition, IJDAR* vol. **26**, pp. 347-362, (2023)<br>
**DOI:** https://doi.org/10.1007/s10032-023-00432-z<br>
**GitHub:** https://github.com/multiscore/e2e-pianoform
