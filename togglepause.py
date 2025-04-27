# Toggle pause.flag file
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAUSE_FLAG = os.path.join(SCRIPT_DIR, "pause.flag")

if os.path.isfile(PAUSE_FLAG):
    os.remove(PAUSE_FLAG)
    print("Removed pause.flag")
else:
    open(PAUSE_FLAG, 'a').close()
    print("Created pause.flag")
