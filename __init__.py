

import os
import sys
import logging

logging.info("[dbpn] run init in dbpn")

## add current dir to sys path so that root dbpn py files can be imported from outside 
current_dir = os.path.dirname( __file__ )
sys.path.append( current_dir )
