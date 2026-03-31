# recovercalc

`recovercalc` is a Python tool for analysing training session files (FIT and similar formats), computing training load and fatigue metrics, and generating simple workout recommendations.

## Features

- FIT file parsing
- Training load metrics (e.g. ATL/CTL/TSB models)
- Fatigue and recovery estimation
- Daily or weekly workout recommendations
- Simple command-line interface

## Usage

Download your last weeks (or all) training activities and store them in `data`.

```bash
pip install -e .
python -m recovercalc.cli
```
You will get some nice graphs and a fit file that should be compatible with GARMIN watches.
Connect your watch with your computer and put the generated file into `NewFiles`.
It will appear now on your watch in workouts for running.

## Status

Early development / experimental.

So far only fit files and GARMIN devices are supported but I want to generalise this.
