#!/bin/zsh
# Shows how much of the iSign video download has arrived.
pid=$(pgrep -f "bin/hf download" | head -1)
if [ -z "$pid" ]; then
  echo "Download is NOT running."
  ls -lh "${0:A:h}"/iSign-videos_v1.1_part_* 2>/dev/null || echo "No finished video parts yet."
  exit
fi
a=$(nettop -P -L 1 -J bytes_in -p $pid 2>/dev/null | tail -1 | cut -d, -f2)
sleep 10
b=$(nettop -P -L 1 -J bytes_in -p $pid 2>/dev/null | tail -1 | cut -d, -f2)
total_mb=$(( b / 1048576 ))
speed=$(( (b - a) / 10 / 1024 ))
echo "Download RUNNING"
echo "Received: $total_mb MB of ~59,000 MB  ($(( total_mb * 100 / 59000 ))%)"
echo "Speed:    $speed KB/s"
[ $speed -gt 0 ] && echo "Time left: about $(( (59000 - total_mb) * 1024 / speed / 3600 )) hours"
