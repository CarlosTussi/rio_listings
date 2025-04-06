'''

    * This is the entry point for the monitoring module.

    * It retrieves the used terminal input with the URL containinig the new dataset.

    * OBS: The dataset needs to be in a .gz file as found at Inside Airbnb. For future versions of this module, flexibility
    of choice will be given to the user, along with input validation.

'''
import sys
from source.monitoring.model_monitoring_report import *

if __name__ == "__main__":
    # Does not procede if user did not provide an input.
    if len(sys.argv) <= 1:
        print("[ERROR] Data URL missing")
    else:
        model_monitoring_report(sys.argv[1])

    