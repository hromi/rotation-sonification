# Rotation Sonification

Sonifies how two HoloLens 2 wearers rotate around their own vertical axis.
Head pose is streamed from [HL2SS](https://github.com/jdibenes/hl2ss) (Spatial Input, port 3812).

| Body | Sound |
|---|---|
| heading | Shepard-tone chroma — 1 turn = 1 octave, clockwise rises forever, counter-clockwise falls forever |
| heading | stereo position |
| angular speed | loudness, brightness, "wind", Leslie-like pulse 4× per turn; stillness = soft hum |
| passing front | bell; pitch walks a pentatonic scale with the winding number |
| two wearers | unison when facing the same way; relative heading = living interval |

## Files

- `hl2ss_launch.py` — start/restart HL2SS via Windows Device Portal, wait for data
- `rotation_sonify.py` — collect head pose, sonify, log to `logs/rotation_<time>.csv`
- `live_graph.py` — live compass / turns / angular-speed window, follows the CSV log
- `hl2ss_si.py` — minimal HL2SS Spatial Input client (stdlib + NumPy)

## Run

```bash
pip install -r requirements.txt
WDP_USER=<user> WDP_PASSWORD=<pass> python hl2ss_launch.py 192.168.1.102 192.168.1.150 --restart
python rotation_sonify.py --hosts 192.168.1.102 192.168.1.150   # r+Enter recenter, q+Enter quit
python live_graph.py                                             # in a second terminal
python rotation_sonify.py --simulate --wav demo.wav              # no hardware needed
```

Each HoloLens has its own world frame: start both wearers facing the same way and recenter (`r`).
Headings are measured from head pose, so whole-body turns are what the instrument expects.
