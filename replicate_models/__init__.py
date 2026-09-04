"""Cog predictors for Replicate: one model per U-MusT task.

    cog.omr.yaml             -> replicate_models/omr.py            (score image -> MusicXML)
    cog.midi-to-audio.yaml   -> replicate_models/midi_to_audio.py  (MIDI -> piano audio)
    cog.image-to-audio.yaml  -> replicate_models/image_to_audio.py (score image -> audio)
    cog.contin-u.yaml        -> replicate_models/contin_u.py       (PDF score -> full audio)

Build and push with `replicate_models/push.sh` (see its header). The
predictors share `replicate_models/common.py`, which points `demo.engine` at
the weights baked into the image under /weights.
"""
