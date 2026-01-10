import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone,xywh_original_percentage
)
class TimeBasedUnauthorizedAccess:
    def __init__(self, parent,zone_data,parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent=parent
        self.parameters = parameters
        self.zone_data = zone_data
        self.running_data = {}
        # Initiating Zone wise
        for zone_name in self.zone_data.keys():
            self.running_data[zone_name]={}
        self.violation_id_data = []
        
        #Initialize Relay
        if parameters["relay"]==1:
            try:
                if self.parent.relay_handler.device==None:
                    success = self.parent.relay_handler.initiate_relay()
                    if not success:
                        print("⚠️ Relay device not available. Continuing without relay control.")
                        self.relay = None
                else:
                    self.relay=self.parent.relay_handler
                    self.switch_relay=parameters["switch_relay"]
            except Exception as e:
                print(f"⚠️ Relay initialization failed: {e}. Continuing without relay control.")
                self.relay = None
        else:
            self.relay = None

        self.last_check_time = self.parameters.get("last_check_time", 0)
        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        # Set up the timezone from the provided string
        self.timezone = pytz.timezone(self.timezone_str)

    def run(self):
        """Main entry point for this activity"""
        """
        Monitor for areas that should have someone present but don't.
        Alerts when no person is detected in a zone for too long.
        """
        print(self.parent.frame_monitor_count)
        if time.time()-self.parameters["last_check_time"]>1:
            self.parameters["last_check_time"]=time.time()
            # Loop through the scheduled times (you may have multiple schedules)
            for schedule in self.parameters["scheduled_time"]:
                # Get the time intervals (start_time, end_time) and days from the current schedule
                time_start_end = schedule["time_start_end"]
                days_of_week = schedule["days"]
                
                # Get current time in the specified timezone
                current_time = datetime.now(self.timezone).time()
                # current_day = datetime.now(timezone).weekday()  # Monday = 0, Sunday = 6
                # Check if today is one of the specified days
                current_day_name = datetime.now(self.timezone).strftime('%A')  # Get the current day as a name (e.g., Monday)
                if current_day_name not in days_of_week:
                    return f"Today is not a valid day for the task. ({current_day_name} is not scheduled.)"
                
                # Loop through all time intervals in the 'time_start_end' list and check if the current time is within any of them
                for time_range in time_start_end:
                    start_time_str, end_time_str = time_range
                    
                    # Convert start and end time strings to time objects
                    start_time = datetime.strptime(start_time_str, "%H:%M").time()
                    end_time = datetime.strptime(end_time_str, "%H:%M").time()
                    
                    # Check if the current time is within the specified time range
                    if start_time <= current_time <= end_time:
                        offender_indices = [i for i, cls in enumerate(self.parent.classes) if cls in self.parameters["subcategory_mapping"]]

                        for idx in offender_indices:
                            box=self.parent.detection_boxes[idx]
                            obj_class=self.parent.classes[idx]
                            tracker_id=self.parent.tracker_ids[idx]
                            anchor = self.parent.anchor_points_original[idx]
                            for zone_name, zone_polygon in self.zone_data.items():
                                if  is_bottom_in_zone(anchor, zone_polygon) and (tracker_id not in self.violation_id_data or self.parameters["relay"]==1):
                                    if tracker_id not in self.running_data[zone_name]:
                                        self.running_data[zone_name][tracker_id] = 1
                                    else:
                                        self.running_data[zone_name][tracker_id]+= 1

                                    if self.running_data[zone_name][tracker_id] > self.parameters["frame_accuracy"]:
                                        if self.relay!=None and self.parameters["relay"]==1:
                                            try:
                                                status=self.relay.state(0)
                                                true_indexes = [(i+1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                                for index in self.switch_relay:
                                                    if (index) not in true_indexes:
                                                        self.relay.state(index, on=True)
                                                    self.relay.start_time[index]=time.time()
                                            except Exception as e:
                                                print(f"⚠️ Relay operation failed: {e}. Continuing without relay control.")
                                        if tracker_id not in self.violation_id_data:
                                            self.violation_id_data.append(tracker_id)
                                            xywh=xywh_original_percentage(box)
                                            datetimestamp=f"{datetime.now(self.timezone).isoformat()}"
                                            self.parent.create_result_events(xywh,obj_class,f"Security-Unauthorized Access",{"zone_name":zone_name},datetimestamp,1,self.parent.image)

        #print("Yes Running Successfully", self.zone_data,self.parameters)

    def cleaning(self):
        self.violation_id_data=[ tracker_id for tracker_id in self.violation_id_data if tracker_id in self.parent.last_n_frame_tracker_ids]
        for zone_name in self.zone_data.keys():
            self.running_data = {
                key: value
                for key, value in self.running_data[zone_name].items()
                if key in self.parent.last_n_frame_tracker_ids
            }
        
