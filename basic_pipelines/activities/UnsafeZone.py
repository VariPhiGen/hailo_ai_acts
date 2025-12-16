import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone,xywh_original_percentage
)

class UnsafeZone:
    def __init__(self, parent,zone_data,parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent=parent
        self.relay=None

        self.parameters = parameters
        self.zone_data = zone_data
        self.running_data = {}

        #Initialize Relay
        if parameters["relay"]==1:
            try:
                if self.parent.relay_handler.device==None:
                    self.parent.relay_handler.initiate_relay()
                self.relay=self.parent.relay_handler
                self.switch_relay=parameters["switch_relay"]
            except Exception:
                self.relay = None
        # Initiating Zone wise
        for zone_name in zone_data.keys():
            self.running_data[zone_name]={}
        self.violation_id_data = []
        
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
        #print(self.parent.frame_monitor_count)
        if time.time()-self.parameters["last_check_time"]>1:
            self.parameters["last_check_time"]=time.time()
            self.relay.check_auto_off(self.switch_relay)
            
            offender_indices = [i for i, cls in enumerate(self.parent.classes) if cls in self.parameters["subcategory_mapping"]]
            print(offender_indices, self.running_data)

            for idx in offender_indices:
                box=self.parent.detection_boxes[idx]
                obj_class=self.parent.classes[idx]
                tracker_id=self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]
                for zone_name, zone_polygon in self.zone_data.items():
                    if is_bottom_in_zone(anchor, zone_polygon) and (tracker_id not in self.violation_id_data or self.parameters["relay"]==1):
                        if tracker_id not in self.running_data[zone_name]:
                            self.running_data[zone_name][tracker_id] = 1
                        else:
                            self.running_data[zone_name][tracker_id]+= 1

                        if self.running_data[zone_name][tracker_id] > self.parameters["frame_accuracy"]:
                            if self.relay!=None and self.parameters["relay"]==1:
                                status=self.relay.state(0)
                                true_indexes = [(i+1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                for index in self.switch_relay:
                                    if (index) not in true_indexes:
                                        self.relay.state(index, on=True)
                                    self.relay.start_time[index]=time.time()
                            if tracker_id not in self.violation_id_data:
                                self.violation_id_data.append(tracker_id)
                                xywh=xywh_original_percentage(box,self.parent.original_width,self.parent.original_height)
                                datetimestamp=f"{datetime.now(self.timezone).isoformat()}"
                                self.parent.create_result_events(xywh,obj_class,f"Hazardous Area-Unsafe Zone",{"zone_name":zone_name},datetimestamp,1,self.parent.image)
            
        #print("Yes Running Successfully", self.zone_data,self.parameters)

    def cleaning(self):
        self.violation_id_data=[ tracker_id for tracker_id in self.violation_id_data if tracker_id in self.parent.last_n_frame_tracker_ids]
        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                key: value
                for key, value in self.running_data[zone_name].items()
                if key in self.parent.last_n_frame_tracker_ids
            }
        
