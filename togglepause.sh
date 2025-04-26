# Toggle pause.flag file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PAUSE_FLAG="$SCRIPT_DIR/pause.flag"

if [ -f "$PAUSE_FLAG" ]; then
  rm "$PAUSE_FLAG"
  echo "Removed pause.flag"
else
  touch "$PAUSE_FLAG"
  echo "Created pause.flag"
fi