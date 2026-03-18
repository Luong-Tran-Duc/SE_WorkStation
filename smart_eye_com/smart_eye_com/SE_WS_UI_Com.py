#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import threading
import ssl
import glob
from typing import Dict, Any
from typing import Optional, Tuple
from datetime import datetime, timezone
import shutil
import itertools
import re
from datetime import datetime

import paho.mqtt.client as mqtt

from minio import Minio
from minio.error import S3Error


# =========================
#  S3 presign + upload part
#  (integrated directly from your existing tool)
# =========================
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROS_ENABLE = True
LASPY_ENABLE = False

if (ROS_ENABLE == True):
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String, Bool
    from stitch_map_msgs.msg import StitchCommand, StitchResults

if (LASPY_ENABLE == True):
    import laspy

#----- testing ----- #
import random
#--------------------#

# =======================================================================
# system Configuration 
# =======================================================================
PROGRAM_DIR = os.path.dirname(os.path.abspath(__file__)) # get path of this program folder

LOG_DIR = os.path.join(os.path.expanduser("~"), "SE_WS_UI_Com_Log")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

BASE_DATA_DIR = os.path.join(os.path.expanduser("~"), "Data_scan")
if not os.path.exists(BASE_DATA_DIR):
    os.makedirs(BASE_DATA_DIR)

MINIO_DATA_DIR = os.path.join(os.path.expanduser("~"), "minio_data", "smarteye-data")

KEEP_FILES_DAY = 7

PROJECT_ID = "SmartEye01"
ENVIRONMENT = "indoor"
NUM_DEVICES = 4

# =======================================================================
# logging system configuration
# =======================================================================
current_log_file = ""
current_log_date = ""
log_lock = threading.Lock()

# =======================================================================
# MinIO Configuration 
# =======================================================================
MINIO_URL = "100.92.74.46:9000"  # IP Tailscale of server
MINIO_ACCESS_KEY = "SmartEye02"          # MinIO username
MINIO_SECRET_KEY = "Viact123"            # MinIO password
MINIO_BUCKET_NAME = "smarteye-data"      # bucket name

# =======================================================================
#                               ROS
# =======================================================================
ROS_NODE_NAME: str = "SE_WS_UI_COM_NODE"

ROS_MSG_SCAN_CMD: str = "/map/command_scan"
ROS_MSG_MAP_STATUS: str = "/map/status"
# only for pantilt status
ROS_MSG_LAS_FILE: str = "/global/las_file"
ROS_MSG_DEVICE_STATUS: str = "/global/device_status"
# =======================================================================
#                                 MQTT
# =======================================================================
MQTT_BROKER: str = "13.213.164.203"
MQTT_PORT: int = 1883
MQTT_USERNAME: str = "admin"
MQTT_PASSWORD: str = "1C5Xag4A7pQb3vCr1eEd8"
MQTT_CERT_PATH = os.path.join(PROGRAM_DIR, "viact2025.crt")
MQTT_USE_TLS: bool = True
MQTT_KEEPALIVE_S: int = 60
# Reconnect/backoff
MQTT_RECONNECT_MIN_S: float = 1.0
MQTT_RECONNECT_MAX_S: float = 60.0
MQTT_RECONNECT_FACTOR: float = 2.0

MQTT_MSG_BASE: str = "/prod/iot/smart_eye"
#---------------------- MQTT message ----------------------#
#------------ SE side ---------------#
MQTT_MSG_SE_CMD_START:     str = f"{MQTT_MSG_BASE}/command/{PROJECT_ID}/Global/start_scan"
MQTT_MSG_SE_CMD_STATUS:    str = f"{MQTT_MSG_BASE}/command/{PROJECT_ID}/Global/check_status"

MQTT_MSG_SE_STATUS_SCAN:   str = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}" # will be + /{DEVICE_ID}/scan_state"
MQTT_MSG_SE_STATUS_DEVICE: str = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}" # will be + /{DEVICE_ID}/device"
MQTT_MSG_SE_STATUS_ERROR:  str = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}" # will be + /{DEVICE_ID}/error"

#------------ UI side ---------------#
MQTT_MSG_UI_CMD_START: str   = f"{MQTT_MSG_BASE}/command/{PROJECT_ID}/start_scan"
MQTT_MSG_UI_CMD_STATUS: str  = f"{MQTT_MSG_BASE}/command/{PROJECT_ID}/check_status"
MQTT_MSG_UI_CMD_SYNC: str    = f"{MQTT_MSG_BASE}/command/{PROJECT_ID}/sync_time"

MQTT_MSG_UI_STATUS_LWT: str  = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/lwt"
MQTT_MSG_UI_STATUS_SCAN: str = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/scan_state"
MQTT_MSG_UI_STATUS_DEVICE: str = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/device"
MQTT_MSG_UI_STATUS_ERROR: str  = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/error"

MQTT_MSG_UI_DATA_LAS: str    = f"{MQTT_MSG_BASE}/data/{PROJECT_ID}/las_file"
MQTT_MSG_UI_TIME_UTC: str    = f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/utc_time"

# =======================================================================
#                              S3 server
# =======================================================================
AI_URL = "https://hi.viact.net/"
PRESIGN_ENDPOINT = "https://hi.viact.net/api/smart-eye/webhook/presigned-s3"
SECRET_ID = "bb238368-2b7e-11ee-be56-0242ac120002"
X_PROJECT_ID_X = "demo_demo_3eb55b6e-01af-4b73-8c9c-70dc9b854a2a"
DEFAULT_DEVICE_ID = "SmartEye01"
PUT_CONTENT_TYPE = "application/octet-stream"
REQUEST_TIMEOUT = 45
forge_model_id = []
FILE_DIR = "https://viact-potreeconverter.s3.ap-east-1.amazonaws.com"

# setup Http connection
def _make_session() -> requests.Session:
    import inspect
    s = requests.Session()

    # Retry common transient errors
    retry_kwargs = dict(
        total=6, connect=6, read=6,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    methods = frozenset(["GET", "POST", "PUT"])
    # urllib3 < 1.26 uses method_whitelist; >= 1.26 uses allowed_methods
    if "allowed_methods" in inspect.signature(Retry).parameters:
        retry_kwargs["allowed_methods"] = methods
    else:
        retry_kwargs["method_whitelist"] = methods  # for older urllib3

    retry = Retry(**retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = _make_session()

# =======================================================================
# logging system helper
# =======================================================================
# write log
def update_log_file(msg, level="INFO"):
    global current_log_file
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {level:7} | {msg}"

    # write to log file
    with log_lock:
        if current_log_file:
            try:
                with open(current_log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry + "\n")
            except Exception as e:
                print(f"Failed to write log to file: {e}")

def clean_old_groups(folder_path, keep_days):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Dictionary to group items by modification date (date only)
    grouped_items = {}

    # Scan all files and folders in the root directory
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Get modification date
        mtime = os.path.getmtime(item_path)
        mod_date = datetime.fromtimestamp(mtime).date()
        
        if mod_date not in grouped_items:
            grouped_items[mod_date] = []
        grouped_items[mod_date].append(item_path)

    # Sort dates from newest to oldest
    sorted_dates = sorted(grouped_items.keys(), reverse=True)

    # Filter out dates to delete (skipping the 'keep_days' newest ones)
    dates_to_delete = sorted_dates[keep_days:]

    # Proceed with deletion
    for d in dates_to_delete:
        for path in grouped_items[d]:
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                    print(f"Deleted file: {path}")
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Deleted folder: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")

# clean old log file
def clean_old_log_thread():
    global current_log_file, current_log_date
    
    while True:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # check if new day
        if today_str != current_log_date:
            with log_lock:
                current_log_date = today_str
                current_log_file = os.path.join(LOG_DIR, f"SE_WS_UI_Com_{today_str}.log")
                
                # create new file if don't have
                if not os.path.exists(current_log_file):
                    with open(current_log_file, "w") as f:
                        f.write(f"--- Log started for {today_str} ---\n")

            # clean old file/folder
            clean_old_groups(LOG_DIR, KEEP_FILES_DAY)
            clean_old_groups(BASE_DATA_DIR, KEEP_FILES_DAY)
            clean_old_groups(MINIO_DATA_DIR, KEEP_FILES_DAY)

        # check each 60s
        time.sleep(60)

# =======================================================================
# system classs
# =======================================================================
class SE_WS_UI_COM(Node):
    # =======================================================================
    # Initiation NODE
    # =======================================================================
    def __init__(self):
        # initial Node ROS 2
        super().__init__(ROS_NODE_NAME)

        global PROJECT_ID
        self.DEVICE_IDs             = [f"Device{i:02d}" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_imu_status      = ["disconnect" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_lidar_status    = ["disconnect" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_camera_status   = ["disconnect" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_pantilt_status  = ["disconnect" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_smarteye_status = ["not_ready" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_scan_status     = ["idle" for i in range(1, NUM_DEVICES + 1)]
        self.DEVICE_las_file_status = ["idle" for i in range(1, NUM_DEVICES + 1)]

        self.DEVICE_las_file_name = ["none" for i in range(1, NUM_DEVICES + 1)]
        
        # check device alive each 60s
        self.DEVICE_last_seen = [time.time() for _ in range(NUM_DEVICES)]

        self.GROUP_imu_status      = "disconnect"
        self.GROUP_lidar_status    = "disconnect"
        self.GROUP_camera_status   = "disconnect"
        self.GROUP_pantilt_status  = "disconnect"
        self.GROUP_smarteye_status = "not_ready"
        self.GROUP_scan_status     = "idle"
        self.GROUP_las_file_status = "idle"

        # subcriber all topic from SE devices with same format ( "+" : a value here)
        self.sub_topics = sub_topics = [
            f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/+/device",
            f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/+/scan_state",
            f"{MQTT_MSG_BASE}/status/{PROJECT_ID}/+/error",
            f"{MQTT_MSG_BASE}/data/{PROJECT_ID}/+/las_file"
        ]

        #---------------------- initiation Minio Client ----------------------#
        self.minio_client = Minio(
            MINIO_URL,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # use HTTP connection via Tailscale
        )

        #----------------------- S3 variables ------------------------#
        self.presigned      = ""
        self.object_key     = ""
        self.forge_model_id = ""

        # safety flag for scan process
        self.scan_start_time = 0.0
        self.SCAN_TIMEOUT_S = 600   # 10 mins
        self.is_waiting_for_files = False
        self.last_scan_status = "idle"

        #---------------------- stitching status and flag ----------------------#
        self.is_stitching = False
        self.stitch_start_time = 0.0
        self.STITCH_TIMEOUT_S = 600.0 # 10 phút
        self.current_scan_folder = ""

        #---------------------- assign Ros Msg ----------------------#
        if ROS_ENABLE:
            # Publisher control START for stitch_node
            self.ros_pub_stitch_cmd = self.create_publisher(StitchCommand, "stitch/command", 10)
            # Subscriber result from stitch_node
            self.ros_sub_stitch_res = self.create_subscription(StitchResults, "stitch/results", self.ros_stitch_results_callback, 1)

        #---------------------- MQTT Setup, connect ----------------------#
        # Initialize the MQTT client instance using the MQTT v3.1.1 protocol
        self.MQTT_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, protocol=mqtt.MQTTv311)

        # Set the authentication credentials (username and password) for the MQTT broker
        self.MQTT_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        # Configure TLS/SSL settings if TLS is enabled and a certificate path is provided
        if MQTT_USE_TLS and MQTT_CERT_PATH:
            # Set the CA certificate for server verification and enforce mandatory certificate requirements
            self.MQTT_client.tls_set(ca_certs=MQTT_CERT_PATH, cert_reqs=ssl.CERT_REQUIRED)

        # Configure automatic reconnection behavior for Paho MQTT library versions 1.6 and above
        if hasattr(self.MQTT_client, "reconnect_delay_set"):
            # Define the exponential backoff delay (minimum and maximum time) between reconnection attempts
            self.MQTT_client.reconnect_delay_set(min_delay=MQTT_RECONNECT_MIN_S, max_delay=MQTT_RECONNECT_MAX_S)

        # Assign callback functions to handle MQTT events
        self.MQTT_client.on_message = self.MQTT_on_message        # Triggered when a message is received
        self.MQTT_client.on_connect = self.MQTT_on_connect        # Triggered when the client connects to the broker
        self.MQTT_client.on_disconnect = self.MQTT_on_disconnect  # Triggered when the client disconnects

        # Initialize state variables and thread-safety tools for managing reconnection logic
        self.MQTT_connected = False                               # Track the current connection status
        self.MQTT_reconnect_lock = threading.Lock()               # Lock to ensure thread-safe operations during reconnection
        self.MQTT_reconnect_thread = None                         # Reference for a separate thread to handle manual reconnection
        self.MQTT_reconnect_delay = MQTT_RECONNECT_MIN_S          # Current delay interval for the next reconnection attempt
        self.MQTT_first_connect = True                            # ignore first auto send start scan command

        self.MQTT_filename = "none"

        self.lifecycle_timer = self.create_timer(60, self.lifecycle_callback)

        if (ROS_ENABLE == True):
            self.get_logger().info(f"[SE_WS_UI_COM_NODE] completed setup")
        else:
            print(f"[SE_WS_UI_COM_NODE] completed setup")
        update_log_file(f"[SE_WS_UI_COM_NODE] completed setup")

    # =======================================================================
    # Helper Functions 
    # =======================================================================
    def _reset_session_states(self):
            # Reset status all devices
            self.DEVICE_scan_status     = ["idle" for _ in range(NUM_DEVICES)]
            self.DEVICE_las_file_status = ["idle" for _ in range(NUM_DEVICES)]
            self.DEVICE_las_file_name   = ["none" for _ in range(NUM_DEVICES)]

            # Reset group status
            self.GROUP_scan_status     = "idle"
            self.GROUP_las_file_status = "idle"
            
            # Reset scan safety status
            self.is_waiting_for_files = False
            self.last_scan_status = "idle"
            self.scan_start_time = 0.0

            msg = "[SE_WS_UI_COM_NODE] All session states have been reset."
            if ROS_ENABLE: self.get_logger().info(msg)
            else: print(msg)
            update_log_file(msg)
            

    def _download_las_files_task(self):

        # 1. create folder with timestamp to save las files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_scan_folder = os.path.join(BASE_DATA_DIR, f"las_file_{timestamp}")
        if not os.path.exists(self.current_scan_folder):
            os.makedirs(self.current_scan_folder)

        local_path = ""

        # 2. download las files from MinIO
        for file_name in self.DEVICE_las_file_name:
            if file_name != "none":
                local_path = os.path.join(self.current_scan_folder, file_name)
                try:
                    msg = f"[SE_WS_UI_COM_NODE] [MinIO] Downloading {file_name} from bucket '{MINIO_BUCKET_NAME}'..."
                    if ROS_ENABLE: self.get_logger().info(msg)
                    else: print(msg)
                    update_log_file(msg)

                    self.minio_client.fget_object(MINIO_BUCKET_NAME, file_name, local_path)
                    
                    success_msg = f"[SE_WS_UI_COM_NODE] [MinIO] Successfully downloaded: {file_name} -> {local_path}"
                    if ROS_ENABLE: self.get_logger().info(success_msg)
                    else: print(success_msg)
                    update_log_file(success_msg)
                except S3Error as e:
                    err_msg = f"[SE_WS_UI_COM_NODE] [MinIO] S3 Error downloading {file_name}: {e}"
                    if ROS_ENABLE: self.get_logger().error(err_msg)
                    else: print(err_msg)
                    update_log_file(err_msg)
                except Exception as e:
                    err_msg = f"[SE_WS_UI_COM_NODE] [MinIO] General Error downloading {file_name}: {e}"
                    if ROS_ENABLE: self.get_logger().error(err_msg)
                    else: print(err_msg)
                    update_log_file(err_msg)

        # 3. start stitching process
        if (NUM_DEVICES > 1):
            log_msg = "[SE_WS_UI_COM_NODE] start stitching process"
            if ROS_ENABLE: self.get_logger().info(log_msg)
            else: print(log_msg)
            update_log_file(log_msg)

            if (ROS_ENABLE == True):
                stitch_cmd = StitchCommand()
                stitch_cmd.command = 1       # START
                stitch_cmd.input_folder = self.current_scan_folder
                stitch_cmd.environment = ENVIRONMENT

                self.ros_pub_stitch_cmd.publish(stitch_cmd)
                self.is_stitching = True
                self.stitch_start_time = time.time()
                self.get_logger().info(f"[SE_WS_UI_COM_NODE] [ROS -> Stitch] Sending: cmd={stitch_cmd.command}, folder={stitch_cmd.input_folder}")
                update_log_file(f"[SE_WS_UI_COM_NODE] [ROS -> Stitch] Sending: cmd={stitch_cmd.command}, folder={stitch_cmd.input_folder}") 
            # emulator stitching process
            else:
                # ---------- testing ---------- #
                time.sleep(30)
                all_files = os.listdir(self.current_scan_folder)
                las_files = [f for f in all_files if f.endswith(".las")]
                if las_files:
                    selected_file = random.choice(las_files)
                    full_path = os.path.join(self.current_scan_folder, selected_file)
                    threading.Thread(target=self._upload_las_files_task, args=(full_path,), daemon=True).start()
                # ------------------------------ #
        else:
            threading.Thread(target=self._upload_las_files_task, args=(local_path,), daemon=True).start()

    def _upload_las_files_task(self, file_input):
        global PROJECT_ID

        if not file_input:
            err_msg = "[SE_WS_UI_COM_NODE] [S3] Invalid file name"
            if ROS_ENABLE: self.get_logger().error(err_msg)
            else: print(err_msg)
            update_log_file(err_msg)
            resp = {"file_name": "", "status": "error", "file_url": "", "forge_model_id": "", "message": "Invalid file name"}
            # update status to UI
            self.MQTT_publish_json(MQTT_MSG_UI_DATA_LAS, resp)
            return

        upload_file_path = file_input
        file_input = os.path.basename(file_input)

        if not os.path.exists(upload_file_path):
            err_msg = "[SE_WS_UI_COM_NODE] [S3] Invalid file path"
            if ROS_ENABLE: self.get_logger().error(err_msg)
            else: print(err_msg)
            update_log_file(err_msg)
            resp = {"file_name": file_input, "status": "error", "file_url": "", "forge_model_id": "", "message": "Invalid file path"}
            # update status to UI
            self.MQTT_publish_json(MQTT_MSG_UI_DATA_LAS, resp)
            return

        # =============================================================
        # update HEADER bound with LASPY
        # =============================================================
        if (LASPY_ENABLE == True):
            try:
                file_name = os.path.basename(file_input)
                msg_repair = f"[SE_WS_UI_COM_NODE] [LASPY] Updating header BBOX for {file_name}..."
                if ROS_ENABLE: self.get_logger().info(msg_repair)
                update_log_file(msg_repair)
    
                # open file LAS
                with laspy.open(upload_file_path) as fh:
                    # read Header and Points
                    las = fh.read()
                    # caculate min/max
                    las.header.update_min_max()
                    # update value Bbox
                    las.write(upload_file_path)
                
                msg_done = f"[SE_WS_UI_COM_NODE] [LASPY] Header updated successfully for {file_name}"
                if ROS_ENABLE: self.get_logger().info(msg_done)
                update_log_file(msg_done)
                
            except Exception as e:
                update_log_file(f"[SE_WS_UI_COM_NODE] [LASPY] Failed to update header: {e}", "ERROR")
            # =============================================================
    
        # 1) Notify uploading
        # update status to UI
        resp = {"file_name": file_input, "status": "uploading", "file_url": "", "forge_model_id": "", "message": "uploading file to S3 server"}
        self.MQTT_publish_json(MQTT_MSG_UI_DATA_LAS, resp)
    
        # 2) Request presigned URL
        ok, msg_ps = self.S3_request_presign(PROJECT_ID)
        if not ok or not self.presigned or not self.object_key:
            err_msg = f"[SE_WS_UI_COM_NODE] [S3] failed at get presigned URL: {msg_ps}"
            if ROS_ENABLE: self.get_logger().error(err_msg)
            else: print(err_msg)
            update_log_file(err_msg)
            resp = {"file_name": file_input, "status": "error", "file_url": "", "forge_model_id": "", "message": "get presigned URL failed"}
            # update status to UI
            self.MQTT_publish_json(MQTT_MSG_UI_DATA_LAS, resp)
            return
    
        # 3) Upload file
        ok2, file_url, msg_up = self.S3_upload_file_with_presigned(upload_file_path)
        if ok2 and file_url:
            time.sleep(3)
            if ROS_ENABLE: self.get_logger().info("uploaded file to S3 server")
            else: print("uploaded file to S3 server")
            update_log_file("uploaded file to S3 server")
            resp = {"file_name": file_input, "status": "completed", "file_url": file_url, "forge_model_id": self.forge_model_id, "message": "upload completed"}
            self.MQTT_publish_json(MQTT_MSG_UI_DATA_LAS, resp)
        else:
            time.sleep(3)
            err_msg = f"[SE_WS_UI_COM_NODE] [S3] failed at push file to S3: {msg_up}"
            if ROS_ENABLE: self.get_logger().error(err_msg)
            else: print(err_msg)
            update_log_file(err_msg)
            resp = {"file_name": file_input, "status": "error", "file_url": "", "forge_model_id": "", "message": "upload file to S3 server failed"}
            # update status to UI
            self.MQTT_publish_json(MQTT_MSG_UI_DATA_LAS, resp)
        # reset scan state
        self._reset_session_states()

    def _group_update_device_status(self):
        # update modules status
        module_buffers = {
            "imu": self.DEVICE_imu_status,
            "lidar": self.DEVICE_lidar_status,
            "camera": self.DEVICE_camera_status,
            "pantilt": self.DEVICE_pantilt_status
        }
        
        for key, buffer in module_buffers.items():
            if any(module_status != "ok" for module_status in buffer):
                setattr(self, f"GROUP_{key}_status", "disconnect")
            else:
                if len(buffer) > 0:
                    setattr(self, f"GROUP_{key}_status", buffer[0])

        # update group_smarteye_status
        if any(SE_status != "ready" for SE_status in self.DEVICE_smarteye_status):
            self.GROUP_smarteye_status = "not_ready"
        else:
            if len(self.DEVICE_smarteye_status) > 0:
                self.GROUP_smarteye_status = self.DEVICE_smarteye_status[0]
    
    def _group_update_scan_state(self):
        # update scan_status
        if any(state in ["failed", "error"] for state in self.DEVICE_scan_status):
            self.GROUP_scan_status = "failed"
        elif all(state == self.DEVICE_scan_status[0] for state in self.DEVICE_scan_status):
            if len(self.DEVICE_scan_status) > 0:
                self.GROUP_scan_status = self.DEVICE_scan_status[0]

        if (self.MQTT_connected == True) and (self.last_scan_status != self.GROUP_scan_status):
            self.last_scan_status = self.GROUP_scan_status
            if self.GROUP_scan_status != "completed":  # scan just completed when las_file completed upload to MinIO server
                self._handle_push_scan_status()

    def _group_update_las_file_status(self, file_name):
        if all(las_status == "completed" for las_status in self.DEVICE_las_file_status):
            self.GROUP_las_file_status = "completed"
            self.is_waiting_for_files = False       # disable wait scan flag
            self._handle_push_scan_status()         # push scan completed
            threading.Thread(target=self._download_las_files_task, daemon=True).start()
        else:
            self.GROUP_las_file_status = "idle"

    # =======================================================================
    # MQTT Handler Functions 
    # =======================================================================
    # ---------------- MQTT wiring ----------------#
    def MQTT_connect_broker(self):
        if(ROS_ENABLE == True):
            self.get_logger().info(f"[SE_WS_UI_COM_NODE] MQTT Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
        else:
            print(f"[SE_WS_UI_COM_NODE] MQTT Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")
        update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Connecting to {MQTT_BROKER}:{MQTT_PORT} ...")

        # try to connect with MQTT brocker
        try:
            self.MQTT_client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE_S)
            self.MQTT_client.loop_start()
        except Exception as e:
            update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Initial connect error: {e}", level="WARN")
            self.MQTT_schedule_reconnect()

    def MQTT_on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.MQTT_connected = True
            self.MQTT_reconnect_delay = MQTT_RECONNECT_MIN_S
            if(ROS_ENABLE == True):
                self.get_logger().info("[SE_WS_UI_COM_NODE] MQTT Connected")
            else:
                print("[SE_WS_UI_COM_NODE] MQTT Connected")
            update_log_file("[SE_WS_UI_COM_NODE] MQTT Connected")
            # subcribe SE MQTT topic
            for SE_topic in self.sub_topics:
                self.MQTT_client.subscribe(SE_topic)
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[SE_WS_UI_COM_NODE] MQTT Subscribed {SE_topic}")
                else:
                    print(f"[SE_WS_UI_COM_NODE] MQTT Subscribed {SE_topic}")
                update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Subscribed {SE_topic}")
            # subcribe UI MQTT topic
            for UI_topic in (MQTT_MSG_UI_CMD_START, MQTT_MSG_UI_CMD_STATUS, MQTT_MSG_UI_CMD_SYNC):
                self.MQTT_client.subscribe(UI_topic)
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[MQTT] Subscribed {UI_topic}")
                else:
                    print(f"[MQTT] Subscribed {UI_topic}")
                update_log_file(f"[MQTT] Subscribed {UI_topic}")
        else:
            if(ROS_ENABLE == True):
                self.get_logger().warn(f"[SE_WS_UI_COM_NODE] MQTT Connect failed rc={rc}")
            else:
                print(f"[SE_WS_UI_COM_NODE] MQTT Connect failed rc={rc}")
            update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Connect failed rc={rc}")
            self.MQTT_schedule_reconnect()

    def MQTT_on_disconnect(self, client, userdata, rc):
        self.MQTT_connected = False
        if(ROS_ENABLE == True):
            self.get_logger().warn(f"[SE_WS_UI_COM_NODE] MQTT Disconnected rc={rc}")
        else:
            print(f"[SE_WS_UI_COM_NODE] MQTT Disconnected rc={rc}")
        update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Disconnected rc={rc}")
        self.MQTT_schedule_reconnect()

    def MQTT_publish_json(self, topic: str, payload: Dict[str, Any]):
        try:
            info = self.MQTT_client.publish(topic, json.dumps(payload))

            if hasattr(info, 'rc') and info.rc != mqtt.MQTT_ERR_SUCCESS:
                if(ROS_ENABLE == True):
                    self.get_logger().warn(f"[SE_WS_UI_COM_NODE] MQTT Publish rc={info.rc} (queued or failed)")
                else:
                    print(f"[SE_WS_UI_COM_NODE] MQTT Publish rc={info.rc} (queued or failed)")
                update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Publish rc={info.rc} (queued or failed)")

            if topic in [MQTT_MSG_UI_STATUS_LWT, MQTT_MSG_UI_STATUS_SCAN, MQTT_MSG_UI_STATUS_DEVICE, MQTT_MSG_UI_STATUS_ERROR, MQTT_MSG_UI_DATA_LAS, MQTT_MSG_UI_TIME_UTC]:
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[SE_WS_UI_COM_NODE] [MQTT -> UI] {topic} {payload}")
                else:
                    print(f"[SE_WS_UI_COM_NODE] [MQTT -> UI] {topic} {payload}")
                update_log_file(f"[SE_WS_UI_COM_NODE] [MQTT -> UI] {topic} {payload}")
            else:
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[SE_WS_UI_COM_NODE] [MQTT -> SE] {topic} {payload}")
                else:
                    print(f"[SE_WS_UI_COM_NODE] [MQTT -> SE] {topic} {payload}")
                update_log_file(f"[SE_WS_UI_COM_NODE] [MQTT -> SE] {topic} {payload}")
        except Exception as e:
            if(ROS_ENABLE == True):
                self.get_logger().error(f"[SE_WS_UI_COM_NODE] MQTT Publish error {e}")
            else:
                print(f"[SE_WS_UI_COM_NODE] MQTT Publish error {e}")
            update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Publish error {e}")

    # ---------------- MQTT message handler ----------------#
    def MQTT_on_message(self, client: mqtt.Client, userdata, message):
        try:
            # devode msg
            text = message.payload.decode() if message.payload else "{}"
            payload = json.loads(text) if text else {}
            topic = message.topic

            # try to fine msg struct: /prod/iot/smart_eye/status/SmartEye01/Device01/device
            parts = topic.split('/')
            # Handle msg from SE deivce
            if len(parts) == 8:
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[SE_WS_UI_COM_NODE] [MQTT <- SE] {message.topic} {payload}")
                else:
                    print(f"[SE_WS_UI_COM_NODE] [MQTT <- SE] {message.topic} {payload}")
                update_log_file(f"[SE_WS_UI_COM_NODE] [MQTT <- SE] {message.topic} {payload}")
                device_id = parts[6] # example: "Device01"
                msg_type = parts[7]  # 'device', 'scan_state', 'las_file'

                # fine index of device (example "Device01" -> index 0)
                try:
                    idx = self.DEVICE_IDs.index(device_id)
                except ValueError:
                    if(ROS_ENABLE == True):
                        self.get_logger().warn(f"[SE_WS_UI_COM_NODE] Unknown Device ID: {device_id}")
                    else:
                        print(f"[SE_WS_UI_COM_NODE] Unknown Device ID: {device_id}")
                    update_log_file(f"[SE_WS_UI_COM_NODE] Unknown Device ID: {device_id}")
                    return

                # update each device state
                if msg_type == "device":
                    self.DEVICE_lidar_status[idx]   = payload.get("lidar", "disconnect")
                    self.DEVICE_imu_status[idx]     = payload.get("imu", "disconnect")
                    self.DEVICE_camera_status[idx]  = payload.get("camera", "disconnect")
                    self.DEVICE_pantilt_status[idx] = payload.get("pantilt", "disconnect")
                    self.DEVICE_smarteye_status[idx]= payload.get("device", "not_ready")
                    self.DEVICE_last_seen[idx] = time.time()  # update last time to check alive
                    # update group module status
                    self._group_update_device_status()

                elif msg_type == "scan_state":
                    self.DEVICE_scan_status[idx] = payload.get("state", "idle")
                    if(self.DEVICE_scan_status[idx] in ["error", "failed"]):
                        self.GROUP_scan_status = "failed"
                        self._handle_push_scan_status()
                        self._reset_session_states()
                    else:
                        self._group_update_scan_state()

                elif msg_type == "las_file":
                    self.DEVICE_las_file_status[idx] = payload.get("status", "idle")
                    self.DEVICE_las_file_name[idx] = payload.get("file_name", "none")
                    self._group_update_las_file_status(self.DEVICE_las_file_name[idx])
                    if self.DEVICE_las_file_status[idx] == "completed":
                        if (ROS_ENABLE == True):
                            self.get_logger().info(f"[SE_WS_UI_COM_NODE] receive Las file from {device_id}: {self.DEVICE_las_file_name[idx]}")
                        else:
                            print(f"[SE_WS_UI_COM_NODE] receive Las file from {device_id}: {self.DEVICE_las_file_name[idx]}")
                        update_log_file(f"[SE_WS_UI_COM_NODE] receive Las file from {device_id}: {self.DEVICE_las_file_name[idx]}")
                    elif self.DEVICE_las_file_status[idx] in ["error", "failed"]:
                        self.GROUP_scan_status = "failed"
                        self._handle_push_scan_status()
                        self._reset_session_states()
            # handle Msg from UI
            else:
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[SE_WS_UI_COM_NODE] [MQTT <- UI] {message.topic} {payload}")
                else:
                    print(f"[SE_WS_UI_COM_NODE] [MQTT <- UI] {message.topic} {payload}")
                update_log_file(f"[SE_WS_UI_COM_NODE] [MQTT <- UI] {message.topic} {payload}")

                if message.topic == MQTT_MSG_UI_CMD_START:
                    self._handle_start_scan(payload)
                elif message.topic == MQTT_MSG_UI_CMD_STATUS:
                    self._handle_check_status()
                elif message.topic == MQTT_MSG_UI_CMD_SYNC:
                    self._handle_sync_time()

        except Exception as e:
            if(ROS_ENABLE == True):
                self.get_logger().error(f"[SE_WS_UI_COM_NODE] MQTT_on_message error: {e}")
            else:
                print(f"[SE_WS_UI_COM_NODE] MQTT_on_message error: {e}")
            update_log_file(f"[SE_WS_UI_COM_NODE] MQTT_on_message error: {e}")

    def _handle_start_scan(self, payload: Dict[str, Any]):
        self.MQTT_filename = str(payload.get("output_name", "none"))
        if self.GROUP_smarteye_status != "ready":
            resp = {"state": "error", "file_name": self.MQTT_filename, "message": "device is not ready"}
            self.MQTT_publish_json(MQTT_MSG_UI_STATUS_SCAN, resp)
        elif (self.GROUP_scan_status == "scanning") and (self.GROUP_smarteye_status == "ready"):
            resp = {"state": "error", "file_name": self.MQTT_filename, "message": "device in previous scan"}
            self.MQTT_publish_json(MQTT_MSG_UI_STATUS_SCAN, resp)
        else:
            self._reset_session_states()
            self.scan_start_time = time.time() # save the start scan time
            self.is_waiting_for_files = True # enable wait scan flag
            self.MQTT_publish_json(MQTT_MSG_SE_CMD_START, payload)
            if(ROS_ENABLE == True):
                self.get_logger().info("[SE_WS_UI_COM_NODE] start scanning point cloud file...")
            else:
                print("[SE_WS_UI_COM_NODE] start scanning point cloud file...")
            update_log_file("[SE_WS_UI_COM_NODE] start scanning point cloud file...")

    def _handle_sync_time(self):
        utc_time = _now_utc_iso()
        resp = {"utc_time": utc_time}
        self.MQTT_publish_json(MQTT_MSG_UI_TIME_UTC, resp)

    def _handle_check_status(self):
        resp = {
            "lidar": self.GROUP_lidar_status,
            "imu": self.GROUP_imu_status,
            "camera": self.GROUP_camera_status,
            "pantilt": self.GROUP_pantilt_status,
            "device": self.GROUP_smarteye_status,
        }
        self.MQTT_publish_json(MQTT_MSG_UI_STATUS_DEVICE, resp)
        self.MQTT_publish_json(MQTT_MSG_UI_STATUS_LWT, {"status": "online", "ts": _now_utc_iso()})

    def _handle_push_scan_status(self):
        if self.GROUP_scan_status == "scanning":
            resp = {"state": "scanning", "file_name": self.MQTT_filename, "message": "device is scanning"}
            self.MQTT_publish_json(MQTT_MSG_UI_STATUS_SCAN, resp)
        elif self.GROUP_scan_status == "completed":
            self.GROUP_scan_status = "idle"
            resp = {"state": "completed", "file_name": self.MQTT_filename, "message": "Scanning completed"}
            self.MQTT_publish_json(MQTT_MSG_UI_STATUS_SCAN, resp)
        elif self.GROUP_scan_status == "failed":
            self.GROUP_scan_status = "idle"
            resp = {"state": "failed", "file_name": self.MQTT_filename, "message": "failed when scanning"}
            self.MQTT_publish_json(MQTT_MSG_UI_STATUS_SCAN, resp)
        elif self.GROUP_scan_status == "error":
            self.GROUP_scan_status = "idle"
            resp = {"state": "error", "file_name": self.MQTT_filename, "message": "error when scanning"}
            self.MQTT_publish_json(MQTT_MSG_UI_STATUS_SCAN, resp)

    #---------- reconnect Helper Functions ----------#
    def MQTT_schedule_reconnect(self):
        """Start a background reconnect thread with exponential backoff if not running."""
        if self.MQTT_connected:
            return
        if getattr(self, "MQTT_reconnect_thread", None) and self.MQTT_reconnect_thread.is_alive():
            return
        with self.MQTT_reconnect_lock:
            if getattr(self, "MQTT_reconnect_thread", None) and self.MQTT_reconnect_thread.is_alive():
                return
            self.MQTT_reconnect_thread = threading.Thread(target=self.MQTT_reconnect_loop, name="mqtt-reconnect", daemon=True)
            self.MQTT_reconnect_thread.start()

    def MQTT_reconnect_loop(self):
        delay = max(MQTT_RECONNECT_MIN_S, getattr(self, "MQTT_reconnect_delay", MQTT_RECONNECT_MIN_S))
        while True and not self.MQTT_connected:
            try:
                if(ROS_ENABLE == True):
                    self.get_logger().warn(f"[SE_WS_UI_COM_NODE] MQTT Reconnecting in {delay:.1f}s...")
                else:
                    print(f"[SE_WS_UI_COM_NODE] MQTT Reconnecting in {delay:.1f}s...")
                update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Reconnecting in {delay:.1f}s...")
                time.sleep(delay)
                self.MQTT_client.reconnect()
                delay = MQTT_RECONNECT_MIN_S  # reset on success path (on_connect will set _connected)
            except Exception as e:
                delay = min(delay * MQTT_RECONNECT_FACTOR, MQTT_RECONNECT_MAX_S)
                self.MQTT_reconnect_delay = delay
                if(ROS_ENABLE == True):
                    self.get_logger().warn(f"[SE_WS_UI_COM_NODE] MQTT Reconnect failed: {e}. Next in {delay:.1f}s")
                else:
                    print(f"[SE_WS_UI_COM_NODE] MQTT Reconnect failed: {e}. Next in {delay:.1f}s")
                update_log_file(f"[SE_WS_UI_COM_NODE] MQTT Reconnect failed: {e}. Next in {delay:.1f}s")
    # =======================================================================
    # Ros message Handler Functions 
    # =======================================================================
    if (ROS_ENABLE == True):
        def ros_stitch_results_callback(self, msg):
            if not self.is_stitching:
                return
            self.get_logger().info(f"[SE_WS_UI_COM_NODE] [ROS <- Stitch] Received: state={msg.state}, path={msg.output_path}, error={msg.error_message}")
            update_log_file(f"[SE_WS_UI_COM_NODE] [ROS <- Stitch] Received: state={msg.state}, path={msg.output_path}, error={msg.error_message}")
            # msg.state == 2 (SUCCESS)
            if msg.state == 2:
                update_log_file(f"[SE_WS_UI_COM_NODE] Stitching SUCCESS. Output: {msg.output_path}")
                self.is_stitching = False

                # trigger task to upload stitch file to S3
                threading.Thread(target=self._upload_las_files_task, args=(msg.output_path,), daemon=True).start()

            elif msg.state == 3: # FAILED
                update_log_file(f"[SE_WS_UI_COM_NODE] Stitching FAILED: {msg.error_message}", level="ERROR")
                self.is_stitching = False
                self.GROUP_scan_status = "failed"
                self._handle_push_scan_status()

        def schedule_stitch_timeout_thread(self):
            while True:
                if self.is_stitching and self.stitch_start_time > 0:
                    if (time.time() - self.stitch_start_time) > self.STITCH_TIMEOUT_S:
                        update_log_file("[SE_WS_UI_COM_NODE] STITCHING TIMEOUT (10 min)!", level="ERROR")
                        self.is_stitching = False
                        self.GROUP_scan_status = "failed"
                        self._handle_push_scan_status()
                time.sleep(10)

    # =======================================================================
    # S3 server Handler Functions 
    # =======================================================================
    def S3_request_presign(self, device_id: str):
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "secret_id": SECRET_ID,
            "x_project_id_x": X_PROJECT_ID_X,
            "user-agent": "SmartEyeUploader/ROS 1.0",
        }
        payload = {"device_id": device_id}

        try:
            resp = _SESSION.post(PRESIGN_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            return False, f"Presign request error: {e}"

        try:
            data = resp.json()
        except ValueError:
            return False, f"Presign returned non-JSON response (status {resp.status_code}): {resp.text[:200]}"

        if not resp.ok:
            return False, f"Presign failed (status {resp.status_code}): {json.dumps(data, ensure_ascii=False)}"

        self.presigned = data.get("presignedUrl")
        self.object_key = data.get("objectKey")
        self.forge_model_id = data.get("forge_model_id")

        if not self.presigned or not self.object_key:
            return False, f"Presign missing fields: {json.dumps(data, ensure_ascii=False)}"

        return True, "ok"

    def S3_upload_file_with_presigned(self, local_path: str):
        if not os.path.isfile(local_path):
            return False, None, f"File not found: {local_path}"

        headers = {"Content-Type": PUT_CONTENT_TYPE}
        try:
            with open(local_path, "rb") as f:
                print(f"presigned: {self.presigned}\n header: {headers} \n data: {f}")
                update_log_file(f"presigned: {self.presigned}\n header: {headers} \n data: {f}")
                resp = _SESSION.put(self.presigned, headers=headers, data=f, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            return False, None, f"Upload connection error: {e}"

        if resp.status_code in (200, 201, 204):
            # Construct the public URL (adjust bucket name if needed)
            file_url = f"{FILE_DIR}/{self.object_key}"
            return True, file_url, "uploaded"
        else:
            body = (resp.text or "").strip()
            if len(body) > 400:
                body = body[:400] + "...(truncated)"
            msg = f"Upload failed HTTP {resp.status_code}. {body}"
            if resp.status_code == 403:
                msg += " (Hint: Content-Type must be correct and URL not expired)"
            return False, None, msg

    # =======================================================================
    # Thread Handler Functions 
    # =======================================================================
    def schedule_update_device_status_thread(self):
        last_update_time = 0.0
        while True:
            now = time.time()
            if (self.MQTT_connected == True) and (now - last_update_time >= 45.0):
                self._handle_check_status()
                last_update_time = time.time()
            time.sleep(45)

    def schedule_daily_scan_thread(self):
        # save last run hour
        last_run_hour = -1  
        while True:
            now = datetime.now() # get current hour
            # check time in 7:00 to 18:00
            if (7 <= now.hour <= 22) and (now.hour != last_run_hour):
                if(ROS_ENABLE == True):
                    self.get_logger().info(f"[SE_WS_UI_COM_NODE] Auto scan trigger at {now.strftime('%H:%M:%S')}")
                else:
                    print(f"[SE_WS_UI_COM_NODE] Auto scan trigger at {now.strftime('%H:%M:%S')}")
                update_log_file(f"[SE_WS_UI_COM_NODE] Auto scan trigger at {now.strftime('%H:%M:%S')}")
                try:
                    if (self.GROUP_smarteye_status != "ready") or (self.GROUP_scan_status == "scanning"):
                        if(ROS_ENABLE == True):
                            self.get_logger().warn("[SE_WS_UI_COM_NODE] Device busy/not ready, skipping auto scan.")
                        else:
                            print("[SE_WS_UI_COM_NODE] Device busy/not ready, skipping auto scan.")
                        update_log_file("[SE_WS_UI_COM_NODE] Device busy/not ready, skipping auto scan.")
                    else:
                        self._reset_session_states()
                        last_run_hour = now.hour
                        resp = {"output_name": f"auto_{now.strftime('%Y%m%d_%H%M%S')}"}
                        self.MQTT_publish_json(MQTT_MSG_SE_CMD_START, resp)
                        self.GROUP_scan_status = "start"
                        if(ROS_ENABLE == True):
                            self.get_logger().info("[SE_WS_UI_COM_NODE] Scan command sent.")
                        else:
                            print("[SE_WS_UI_COM_NODE] Scan command sent.")
                        update_log_file("[SE_WS_UI_COM_NODE] Scan command sent.")
                except Exception as e:
                    if(ROS_ENABLE == True):
                        self.get_logger().error(f"[SE_WS_UI_COM_NODE] Error executing scan: {e}")
                    else:
                        print(f"[SE_WS_UI_COM_NODE] Error executing scan: {e}")
                    update_log_file(f"[SE_WS_UI_COM_NODE] Error executing scan: {e}")
            # check every 30s.
            time.sleep(30)

    # check device alive
    def schedule_check_device_timeout_thread(self):
        TIMEOUT_LIMIT = 60.0 
        while True:
            now = time.time()
            has_changed = False

            for idx in range(len(self.DEVICE_IDs)):
                if now - self.DEVICE_last_seen[idx] > TIMEOUT_LIMIT:
                    # Reset all status
                    self.DEVICE_lidar_status[idx]   = "disconnect"
                    self.DEVICE_imu_status[idx]     = "disconnect"
                    self.DEVICE_camera_status[idx]  = "disconnect"
                    self.DEVICE_pantilt_status[idx] = "disconnect"
                    self.DEVICE_smarteye_status[idx]= "not_ready"

                    has_changed = True
                    msg = f"[SE_WS_UI_COM_NODE] Device {self.DEVICE_IDs[idx]} lost connection (Timeout 60s)"
                    if ROS_ENABLE: self.get_logger().warn(msg)
                    else: print(msg)
                    update_log_file(msg)

            # update grou status
            if has_changed:
                self._group_update_device_status()

            # check each 20s
            time.sleep(20)

    # handler timeout of scan process
    def schedule_scan_timeout_thread(self):
        while True:
            now = time.time()
            if self.is_waiting_for_files and self.scan_start_time > 0:
                if (now - self.scan_start_time) > self.SCAN_TIMEOUT_S:
                    # check missing file device
                    missing_devices = [self.DEVICE_IDs[i] for i, status in enumerate(self.DEVICE_las_file_status) if status != "completed"]

                    msg = f"[SE_WS_UI_COM_NODE] SCAN TIMEOUT! Missing files from: {missing_devices}"
                    if ROS_ENABLE: self.get_logger().error(msg)
                    else: print(msg)
                    update_log_file(msg)

                    # scan process failed
                    self.GROUP_scan_status = "failed"
                    self.is_waiting_for_files = False

                    self._handle_push_scan_status()

                    # Reset for next scan
                    self.scan_start_time = 0.0
            # check each 20s
            time.sleep(20)

    #---------- testing ----------#
    def schedule_test_scan_thread(self):
        if ROS_ENABLE: 
            self.get_logger().info("[SE_WS_UI_COM_NODE] Test thread started: Trigger scan every 15 minutes.")
        else: 
            print("[SE_WS_UI_COM_NODE] Test thread started: Trigger scan every 15 minutes.")
        update_log_file("[SE_WS_UI_COM_NODE] Test thread started: Trigger scan every 15 minutes.")

        while True:
            # trigger start scan each 15 min
            time.sleep(15 * 60)
            
            now = datetime.now()
            test_output_name = f"test_{now.strftime('%Y%m%d_%H%M%S')}"
            
            try:
                self._reset_session_states()
                
                payload = {"output_name": test_output_name}
                self.MQTT_publish_json(MQTT_MSG_SE_CMD_START, payload)
                
                self.GROUP_scan_status = "start"
                
                msg = f"[TEST TRIGGER] Scan command sent: {test_output_name}"
                if ROS_ENABLE: self.get_logger().info(msg)
                else: print(msg)
                update_log_file(msg)
                
            except Exception as e:
                err_msg = f"[TEST TRIGGER] Error during auto trigger: {e}"
                if ROS_ENABLE: self.get_logger().error(err_msg)
                else: print(err_msg)
                update_log_file(err_msg)
        #---------- testing ----------#

    # ---------------- lifecycle ----------------
    def lifecycle_callback(self):
        msg = "[SE_WS_UI_COM_NODE] node alive check"
        if ROS_ENABLE:
            self.get_logger().info(msg)
        else:
            print(msg)

        if not self.MQTT_connected:
            self.MQTT_schedule_reconnect()

    # need MinIO server work before join system
    def wait_for_minio_ready(self):
        msg = f"[SE_WS_UI_COM_NODE] Checking MinIO server at {MINIO_URL}..."
        if ROS_ENABLE: self.get_logger().info(msg)
        else: print(msg)
        update_log_file(msg)

        while True:
            try:
                # check bucket available
                if self.minio_client.bucket_exists(MINIO_BUCKET_NAME):
                    success_msg = "[SE_WS_UI_COM_NODE] MinIO server is ONLINE and bucket is ready."
                    if ROS_ENABLE: self.get_logger().info(success_msg)
                    else: print(success_msg)
                    update_log_file(success_msg)
                    return True
                else:
                    warn_msg = f"[SE_WS_UI_COM_NODE] MinIO is online but bucket '{MINIO_BUCKET_NAME}' not found. Retrying..."
                    if ROS_ENABLE: self.get_logger().warn(warn_msg)
                    else: print(warn_msg)
                    update_log_file(warn_msg)
            except Exception as e:
                err_msg = f"[SE_WS_UI_COM_NODE] MinIO server is OFFLINE or unreachable. Retrying in 5s... (Error: {e})"
                if ROS_ENABLE: self.get_logger().warn(err_msg)
                else: print(err_msg)
                update_log_file(err_msg)
            # check each 5 seconds
            time.sleep(5)

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def main(args=None):
    # create log file
    global current_log_date, current_log_file
    current_log_date = datetime.now().strftime("%Y-%m-%d")
    current_log_file = os.path.join(LOG_DIR, f"SE_WS_UI_Com_{current_log_date}.log")
    if not os.path.exists(current_log_file):
        with open(current_log_file, "w") as f:
            f.write(f"[SE_WS_UI_COM_NODE] --- Log started for {current_log_date} ---\n")

    # initial rclpy and system class
    rclpy.init(args=args)
    system = SE_WS_UI_COM()

    system.get_logger().info("[SE_WS_UI_COM_NODE] run version 2.0")

    if(ROS_ENABLE == True):
        system.get_logger().info("[SE_WS_UI_COM_NODE] node ready")
    else:
        print("[SE_WS_UI_COM_NODE] node ready")
    update_log_file("[SE_WS_UI_COM_NODE] node ready")

    # wait MinIO server
    system.wait_for_minio_ready()

    # connect MQTT brocker
    system.MQTT_connect_broker()

    # Initialize multithreading for ROS 2
    executor = MultiThreadedExecutor()
    executor.add_node(system)

    # start all thread handler
    threading.Thread(target=clean_old_log_thread, daemon=True).start()
    time.sleep(60)

    threading.Thread(target=system.schedule_check_device_timeout_thread, daemon=True).start()
    threading.Thread(target=system.schedule_update_device_status_thread, daemon=True).start()
    threading.Thread(target=system.schedule_scan_timeout_thread, daemon=True).start()
    threading.Thread(target=system.schedule_daily_scan_thread, daemon=True).start()

    if ROS_ENABLE:
        threading.Thread(target=system.schedule_stitch_timeout_thread, daemon=True).start()

    #---------- testing ----------#
    # threading.Thread(target=system.schedule_test_scan_thread, daemon=True).start()
    #---------- testing ----------#

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        system.MQTT_client.loop_stop()
        system.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    print("run version 2.0")
    main()