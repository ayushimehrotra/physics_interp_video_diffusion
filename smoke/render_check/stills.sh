#!/bin/bash
# stills.sh <compare.mp4> <out.png> <frame list...>: one tall strip of the given frames
src=$1; out=$2; shift 2; sel=$(printf "eq(n\\\\,%s)+" "$@"); sel=${sel%+}
ffmpeg -v error -y -i "$src" -vf "select='$sel',scale=iw*min(1\,1100/iw):-2,tile=1x$#" -frames:v 1 "$out"
