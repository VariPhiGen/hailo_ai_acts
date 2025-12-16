import time
from datetime import datetime
import pytz
from activities.activity_helper_utils import (
    is_bottom_in_zone,xywh_original_percentage
)
from shapely.geometry import Point, Polygon

class PPE:
    def __init__(self, parent,zone_data,parameters):
        """
        parent: reference to user_app_callback_class (for detections, events, etc.)
        """
        self.parent=parent

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

        # Initiating Acts wise
        self.violation_id_data = {}
        for acts in parameters["subcategory_mapping"]:
            self.violation_id_data[acts]=[]
        
        self.last_check_time = self.parameters.get("last_check_time", 0)
        self.timezone_str = self.parameters.get("timezone", "Asia/Kolkata")
        # Set up the timezone from the provided string
        self.timezone = pytz.timezone(self.timezone_str)

    def run(self):
        #print(self.parent.frame_monitor_count)
        if time.time()-self.parameters["last_check_time"]>1:
            ppe_objects=self.parameters["subcategory_mapping"]
            person_indices = [i for i, cls in enumerate(self.parent.classes) if cls == "person"]
            ppe_indices = [i for i, cls in enumerate(self.parent.classes) if cls in ppe_objects.keys()]
            self.parameters["last_check_time"]=time.time()

            for idx in person_indices:
                box=self.parent.detection_boxes[idx]
                obj_class=self.parent.classes[idx]
                tracker_id=self.parent.tracker_ids[idx]
                anchor = self.parent.anchor_points_original[idx]
                for zone_name, zone_polygon in self.zone_data.items():
                    if is_bottom_in_zone(anchor, zone_polygon):
                        person_poly = Polygon([(box[0], box[1]), (box[0], box[3]), (box[2], box[3]),  (box[2], box[1])])
                        for ppe_idx in ppe_indices:
                            ppe_box=self.parent.detection_boxes[ppe_idx]
                            ppe_obj_class=self.parent.classes[ppe_idx]
                            if self.is_object_in_zone(ppe_box,person_poly) and (tracker_id not in self.violation_id_data[ppe_obj_class] or self.parameters["relay"]==1):
                                if tracker_id not in self.running_data[zone_name]:
                                    self.running_data[zone_name][tracker_id] = {}
                                # Now, check if the ppe_obj_class key exists for this tracker_id
                                if ppe_obj_class not in self.running_data[zone_name][tracker_id]:
                                    self.running_data[zone_name][tracker_id][ppe_obj_class] = 1
                                else:
                                    self.running_data[zone_name][tracker_id][ppe_obj_class] += 1

                                if self.running_data[zone_name][tracker_id][ppe_obj_class] > self.parameters["frame_accuracy"]:
                                    if self.relay!=None and self.parameters["relay"]==1:
                                        status=self.relay.state(0)
                                        true_indexes = [(i+1) for i, x in enumerate(status) if isinstance(x, bool) and x is True]
                                        for index in self.switch_relay:
                                            if (index) not in true_indexes:
                                                self.relay.state(index, on=True)
                                            self.relay.start_time[index]=time.time()
                                    if tracker_id not in self.violation_id_data[ppe_obj_class]:
                                        self.violation_id_data[ppe_obj_class].append(tracker_id)
                                        xywh=xywh_original_percentage(box,self.parent.original_width,self.parent.original_height)
                                        datetimestamp=f"{datetime.now(self.ist_timezone).isoformat()}"
                                        subcategory=ppe_objects[ppe_obj_class]
                                        self.create_result_events(xywh,obj_class,f"PPE-{subcategory}",{"zone_name":zone_name},datetimestamp,1,self.parent.image)

    def cleaning(self):
        self.violation_id_data=[ tracker_id for tracker_id in self.violation_id_data if tracker_id in self.parent.last_n_frame_tracker_ids]
        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                key: value
                for key, value in self.running_data[zone_name].items()
                if key in self.parent.last_n_frame_tracker_ids
            }
        for acts in self.parameters["subcategory_mapping"]:
            self.violation_id_data[acts]=[ tracker_id for tracker_id in self.violation_id_data[acts] if tracker_id in self.parent.last_n_frame_tracker_ids]
        for zone_name in self.zone_data.keys():
            self.running_data[zone_name] = {
                key: value
                for key, value in self.running_data[zone_name].items()
                if key in self.parent.last_n_frame_tracker_ids
            }
        
