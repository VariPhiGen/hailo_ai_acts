import serial
import time
from collections import deque
from threading import Thread, Lock
from typing import Optional, Dict, Any, List, Tuple


class RadarHandler:
    """Handles radar communication and speed data processing."""
    
    def __init__(self):
        """Initialize radar handler."""
        self.radar_port = None  # Serial port for radar
        self.radar_baudrate = 9600  # Default baudrate
        self.radar_thread = None
        self.radar_running = False
        self.radar_lock = Lock()  # Lock for thread-safe radar data access
        self.count_radar = 0  # Count radar until it will become 0 again
        # Use deques for better performance with time-series data
        # self.rank1_radar_speeds = deque()
        self.rank2_radar_speeds = deque()
        self.rank3_radar_speeds = deque()
        self.rankl_radar_speeds = deque()
        self.latest_radar_speed = deque(maxlen=1)
        self.flag=0
        self.ser = None
        self.is_calibrating = {}
        
        # Connectivity monitoring
        self.last_successful_read = time.time()
        self.connection_timeout = 30  # Seconds without data before considering disconnected
        self.reconnect_interval = 5  # Seconds to wait between reconnection attempts
        self.max_reconnect_attempts = 3  # Maximum reconnection attempts
        self.reconnect_attempts = 0
        self.is_connected = False
        
    def init_radar(self, port: str, baudrate: int = 9600, max_age=10, max_diff_rais=15, calibration_required=2):
        """
        Initialize radar connection parameters.
        
        Args:
            port: Serial port for radar connection
            baudrate: Baud rate for serial communication
            max_age: Maximum age of speed readings in seconds
            max_diff_rais: Maximum difference between consecutive readings
            calibration_required: Number of calibration readings required
        """
        self.radar_port = port
        self.radar_baudrate = baudrate
        self.max_age = max_age
        self.max_diff_rais = max_diff_rais
        self.calibration_required = calibration_required
        self.class_calibration_count = {}  # Track calibration count per class
        self.ser = None
        self._connect()
        
    def start_radar(self):
        """Start the radar reading thread."""
        if not self.radar_running and self.radar_port:
            self.radar_running = True
            self.radar_thread = Thread(target=self._radar_read_loop, daemon=True)
            self.radar_thread.start()

    def stop_calbirating(self,obj_class):
        self.is_calibrating[obj_class]=False
            
    def stop_radar(self):
        """Stop the radar reading thread."""
        self.radar_running = False
        if self.radar_thread:
            self.radar_thread.join()
    
    def _connect(self) -> bool:
        """Establish serial connection to radar."""
        try:
            if self.ser:
                self.ser.close()
            
            self.ser = serial.Serial(
                port=self.radar_port,
                baudrate=self.radar_baudrate,
                timeout=0.02,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            print(f"Radar connected successfully to {self.radar_port}")
            return True
        except Exception as e:
            self.is_connected = False
            error_msg = f"Failed to connect to radar at {self.radar_port}: {e}"
            print(error_msg)
            if hasattr(self, 'error_logger') and self.error_logger:
                self.error_logger(error_msg)
            return False
    
    def _check_connectivity(self) -> bool:
        """Check if radar is still connected and responding."""
        if not self.ser or not self.ser.is_open:
            return False
        
        # Check if we've received data recently
        if time.time() - self.last_successful_read > self.connection_timeout:
            return False
        
        return True
    
    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to radar."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            error_msg = f"Max reconnection attempts ({self.max_reconnect_attempts}) reached for radar"
            if hasattr(self, 'error_logger') and self.error_logger:
                self.error_logger(error_msg)
            return False
        
        self.reconnect_attempts += 1
        print(f"Attempting radar reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        if self._connect():
            return True
        
        # Wait before next attempt
        time.sleep(self.reconnect_interval)
        return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current radar connection status."""
        return {
            'is_connected': self.is_connected,
            'port': self.radar_port,
            'last_successful_read': self.last_successful_read,
            'reconnect_attempts': self.reconnect_attempts,
            'time_since_last_read': time.time() - self.last_successful_read if self.last_successful_read else None
        }
    
    def reset_connection(self):
        """Reset connection state and attempt reconnection."""
        self.is_connected = False
        self.reconnect_attempts = 0
        self.last_successful_read = time.time()
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
            self.ser = None
        return self._connect()
            
    def set_error_logger(self, error_logger):
        """Set the error logger function for sending errors to Kafka."""
        self.error_logger = error_logger
    
    def _process_speed_data(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Process raw radar speed data.
        
        Args:
            data: Raw bytes from radar
            
        Returns:
            Processed speed data dictionary or None
        """
        if not data:
            return None
            
        try:
            # Convert bytes to list of hex strings for debugging
            hex_data = [hex(x) for x in data]
            # Debug data removed - can be added back if needed
            
            # Process each 4-byte chunk
            for i in range(len(data) - 3):
                # Check for target speed pattern: 0xFC 0xFA sum 0x00
                if data[i] == 0xFC and data[i+1] == 0xFA and data[i+3] == 0x00:
                    speed_raw = data[i+2]
                    if 0x0F <= speed_raw <= 0xFA:  # Valid speed range
                        speed_kmh = speed_raw
                        direction = 'Approaching'
                        return {
                            'speed': speed_kmh,
                            'direction': direction,
                            'type': 'Primary Target'
                        }
                
                # Check for leading target speed pattern: 0xFB 0xFD sum 0x00
                elif data[i] == 0xFB and data[i+1] == 0xFD and data[i+3] == 0x00:
                    speed_raw = data[i+2]
                    if 0x00 <= speed_raw <= 0xFA:  # Valid speed range
                        speed_kmh = speed_raw
                        direction = 'Receding'
                        
                        return {
                            'speed': speed_kmh,
                            'direction': direction,
                            'type': 'Leading Target'
                        }
            
            return None
        except Exception as e:
            if hasattr(self, 'error_logger') and self.error_logger:
                self.error_logger(f"Error processing speed data: {e}")
            return None

    def get_speed(self) -> Optional[Dict[str, Any]]:
        """
        Get current speed reading from radar.
        Fully fault-tolerant: auto-reconnects if disconnected.
        """
        # Ensure serial connection exists
        if not self._check_connectivity():
            self.is_connected = False
            self.ser = None
            self._attempt_reconnection()
            if not self.is_connected:
                return None

        if not self.ser or not self.ser.is_open:
            return None

        try:
            data = self.ser.read(4)  # read 4 bytes
            if len(data) == 4:
                self.last_successful_read = time.time()
                return self._process_speed_data(data)
        except serial.SerialException as e:
            if hasattr(self, 'error_logger') and self.error_logger:
                self.error_logger(f"Radar read error: {e}")
            self.is_connected = False
            self.ser = None  # force reconnection
        return None

    
    def _radar_read_loop(self):
        """
        Main radar reading loop running in a separate thread.
        This version is fully continuous, fault-tolerant, and avoids busy-waiting.
        """
        previous_reading = 0
        last_connectivity_check = time.time()

        while self.radar_running:
            current_time = time.time()

            # Periodic connectivity check every 10 seconds
            if current_time - last_connectivity_check > 10:
                if not self._check_connectivity():
                    print("Radar disconnected - attempting reconnection")
                    success = self._attempt_reconnection()
                    if not success:
                        if hasattr(self, 'error_logger') and self.error_logger:
                            self.error_logger("Radar reconnection failed, will retry continuously")
                last_connectivity_check = current_time

            # Try reading radar speed
            speed_data = self.get_speed()
            if speed_data is None:
                time.sleep(0.05)  # small delay to prevent busy loop
                continue

            speed = speed_data['speed']
            direction = speed_data['direction']
            target_type = speed_data['type']

            # Thread-safe processing
            with self.radar_lock:
                current_time = time.time()
                if previous_reading != 0 and abs(speed - previous_reading) >= 4:
                    self.count_radar = 0
                    if self.flag == 0:
                        self._add_speed_to_rank(previous_reading, self.rankl_radar_speeds, current_time)
                    else:
                        self.flag = 0
                previous_reading = speed

                self.latest_radar_speed.append((current_time, speed))
                if speed != 0:
                    self.count_radar += 1
                    # Handle rank logic
                    if self.count_radar == 1:
                        self._cleanup_old_speeds(self.rankl_radar_speeds, current_time)
                    elif self.count_radar != 0:
                        self._add_speed_to_rank(speed, self.rank3_radar_speeds, current_time)
                        self._process_rank3_to_rank2()
                    # print(f"DEBUG: Latest speed: {self.latest_radar_speed}, Rankl: {self.rankl_radar_speeds}")
        
    def get_radar_data(self, ai_speed,threshold, obj_class):
        """
        Optimized radar speed matching with early exit and cleaner logic
        """
        with self.radar_lock:
            # Early exit if no radar data available
            if not any([self.rankl_radar_speeds, self.rank2_radar_speeds, self.rank3_radar_speeds]):
                return None,False
            
            # Filter speeds above threshold once
            min_speed = threshold - 2
            
            # Process calibration mode
            if self.is_calibrating[obj_class]:
                rs,r1=self._handle_calibration_mode(ai_speed, obj_class, min_speed)
                return rs,r1
            # Normal mode: try each rank in order
            rs,r1=self._get_best_match_from_ranks(ai_speed, min_speed)
            return rs,r1

    def _handle_calibration_mode(self, ai_speed, obj_class, min_speed):
        """
        Handle calibration mode with early exit
        """
        # Filter rank1 speeds once - more efficient with list comprehension
        result=self.rankl_radar_speeds+self.latest_radar_speed
        valid_speeds = [(ts, speed) for ts, speed in result if speed > min_speed]
        if not valid_speeds or len(valid_speeds)>1:
            return None,False
        
        # Find best match
        best_match = min(valid_speeds, key=lambda x: abs(x[1] - ai_speed))
        
        # Update calibration count
        if self.class_calibration_count[obj_class] < self.calibration_required:
            self.class_calibration_count[obj_class] += 1
            print(f"Calibration for {obj_class}: {self.class_calibration_count[obj_class]}/{self.calibration_required} done.")
        else:
            self.stop_calbirating(obj_class)
            
        
        # Remove used speed from the correct deque
        try:
            if best_match in self.rankl_radar_speeds:
                self.rankl_radar_speeds.remove(best_match)
            elif best_match in self.latest_radar_speed:
                self.latest_radar_speed.remove(best_match)
                self.flag=1
        except ValueError:
            # If best_match is not found in either deque, continue without error
            pass
        return best_match[1],True

    def _get_best_match_from_ranks(self, ai_speed, min_speed):
        """
        Get best match from available ranks with early exit
        """
        # Define ranks to check in order of priority
        rank_configs = [
            [self.rankl_radar_speeds, 'rank1'],
            [self.rank3_radar_speeds, 'rank3'], 
            [self.rank2_radar_speeds, 'rank2']
        ]
        rank1=False
        for radar_speeds, rank_name in rank_configs:
            # Filter speeds above threshold - more efficient with list comprehension
             # For latest Speed
            if rank_name == "rank1":
                if len(self.latest_radar_speed) > 0 and int(self.latest_radar_speed[0][1]) != 0:
                    result = radar_speeds + self.latest_radar_speed
                else:
                    result = radar_speeds   

                #print("Result fron rankl Best Match",result)
                valid_speeds = [(ts, speed) for ts, speed in result]
            else:
                valid_speeds = [(ts, speed) for ts, speed in radar_speeds if (speed-ai_speed) < self.max_diff_rais and speed > min_speed]
            # print(radar_speeds,rank_name)
            if valid_speeds:
                # Get earliest timestamp (best match)
                best_match = min(valid_speeds, key=lambda x: abs(x[1] - ai_speed))
                
                if len(valid_speeds)==1 and rank_name == "rank1":
                  rank1=True
                  
				# Remove used speed
                try:
                    if best_match in radar_speeds:
                        radar_speeds.remove(best_match)
                    elif best_match in radar_speeds:
                        radar_speeds.remove(best_match)
                        self.flag=1
                except ValueError:
                    # If best_match is not found in either deque, continue without error
                    pass
                return best_match[1],rank1
        
        # No valid speeds found in any rank
        return None,rank1
        
    def _cleanup_old_speeds(self, speed_deque: deque, current_time: float) -> None:
        """
        Remove old speed entries from a deque based on max_age.
        
        Args:
            speed_deque: Deque containing (timestamp, speed) tuples
            current_time: Current timestamp
        """
        while speed_deque and (current_time - speed_deque[0][0]) >= self.max_age:
            speed_deque.popleft()
    
    def _add_speed_to_rank(self, speed: int, rank_deque: deque, current_time: float) -> None:
        """
        Add speed to a rank deque with cleanup.
        
        Args:
            speed: Speed value to add
            rank_deque: Target deque to add to
            current_time: Current timestamp
        """
        self._cleanup_old_speeds(rank_deque, current_time)
        rank_deque.append((current_time, speed))
    
    def _process_rank3_to_rank2(self) -> None:
        """
        Process rank2 speeds and move oldest to rank2 when rank3 has 2 entries.
        """
        if len(self.rank3_radar_speeds) >= 2:
            # Move oldest speed from rank3 to rank2
            oldest_speed = self.rank3_radar_speeds.popleft()
            self.rank2_radar_speeds.append(oldest_speed)
            
            # Keep only the most recent entry in rank3
            if len(self.rank2_radar_speeds) > 1:
                self.rank2_radar_speeds.popleft()