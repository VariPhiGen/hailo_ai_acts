import time
from datetime import datetime
import pytz

class TimeBasedUnauthorizedAccess:
    def __init__(self, parent,zone_data,parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent = parent
        self.parameters = parameters
        self.zone_data = zone_data
        self.violation_id_data = []
        self.TBUA_data = {}
        self.last_check_time = self.parameters.get("last_check_time", 0)

    def run(self):
        """Main entry point for this activity"""
        print("Yes Running Successfully", self.zone_data,self.parameters)
        
