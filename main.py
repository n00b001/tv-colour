import logging
import multiprocessing
import time
from datetime import datetime, timedelta

import coloredlogs
import cv2
import numpy as np
import requests  # Import requests for API calls
from plexapi.server import PlexServer

from config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

# Cache to store video file connections and last access time
video_cache = {}
session_cache = {}  # Cache for session.guid to video path
cache_lock = multiprocessing.Lock()  # Use multiprocessing lock


class PlexMonitor:
    def __init__(self, plex_server_url, plex_token, ha_url, ha_entity_id):
        self.plex = PlexServer(plex_server_url, plex_token)
        self.ha_url = ha_url
        self.ha_entity_id = ha_entity_id
        self.loop_times = []
        self.loop_start_time = time.time()
        self.last_frame_log_time = 0  # Track last log time for frame retrieval failure
        self.last_avg_color = None
        self.last_light_color = None  # Add this line to track the last light color

    def background_cache_cleanup(self):
        """Background process that removes old cache items every hour."""
        while True:
            time.sleep(3600)  # Run the cleanup every hour
            self.clean_old_cache_items()

    def clean_old_cache_items(self):
        """Remove video cache items that haven't been accessed in the last 12 hours."""
        current_time = datetime.now()

        with cache_lock:
            items_to_remove = []
            # Find all items that are older than 12 hours in video cache
            for video_path, (cap, last_access_time) in video_cache.items():
                if current_time - last_access_time > timedelta(hours=CACHE_EXPIRY_HOURS):
                    items_to_remove.append(video_path)

            # Remove outdated video cache items and release the video capture objects
            for video_path in items_to_remove:
                logger.info(f"Removing cached video: {video_path}")
                video_cache[video_path][0].release()  # Release the cv2.VideoCapture object
                del video_cache[video_path]

            items_to_remove = []
            # Clean session cache based on GUIDs
            for guid in list(session_cache.keys()):
                if current_time - session_cache[guid][1] > timedelta(hours=CACHE_EXPIRY_HOURS):
                    items_to_remove.append(guid)

            for guid in items_to_remove:
                logger.info(f"Removing cached session GUID: {guid}")
                del session_cache[guid]

    def get_average_color(self, video_path, playback_time_milliseconds, loop_offset):
        """Retrieve the average color of the frame at the given playback time."""
        with cache_lock:
            # Check if the video is already cached
            if video_path not in video_cache:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None
                video_cache[video_path] = (cap, datetime.now())  # Cache the video connection
            else:
                cap, _ = video_cache[video_path]
                video_cache[video_path] = (cap, datetime.now())  # Update access time

            # Set the frame position to the desired playback time
            # Set to specific playback time in milliseconds
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                playback_time_milliseconds + loop_offset + TIME_TO_SET_LIGHT_MS
            )

            # Read the frame at that specific time
            ret, frame = cap.read()

        if not ret:
            return None

        # Calculate the average color of the frame (in BGR format)
        # avg_color_rgb = np.average(np.average(frame, axis=0), axis=0)[::-1]  # Convert BGR to RGB
        avg_color_rgb = np.mean(frame, axis=(0, 1))[::-1]
        return avg_color_rgb

    def rgb_to_xy(self, r, g, b):
        """Convert RGB to CIE 1931 XY color space."""
        # Apply gamma correction
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

        r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
        g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
        b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

        # Convert to XYZ space
        X = r * 0.4124 + g * 0.3576 + b * 0.1805
        Y = r * 0.2126 + g * 0.7152 + b * 0.0722
        Z = r * 0.0193 + g * 0.1192 + b * 0.9505

        # Convert to xy coordinates
        if (X + Y + Z) == 0:
            return 0, 0
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)

        return x, y

    def set_light_color(self, color, average_loop_time_milliseconds):
        """Set the color of the lights in Home Assistant with a transition using XY color space and adjust brightness."""
        # Convert RGB to XY
        xy_color = self.rgb_to_xy(*color)

        # Calculate brightness based on the luminance value (Y in the RGB to XYZ conversion)
        brightness = int(max(color) / 255.0 * 100)  # Normalized to a percentage scale (0-100)

        # Prepare payload for the Home Assistant API call
        payload = {
            "entity_id": self.ha_entity_id,
            "xy_color": list(xy_color),  # Convert to list for API
            "brightness_pct": brightness,  # Adjust brightness (0-100%)
            "transition": (average_loop_time_milliseconds / 1000) * 8,  # Adjust transition as needed
        }

        # Check if the color is different from the last set color
        if self.last_light_color is None or not np.array_equal(xy_color, self.last_light_color):
            # Set the headers with the access token
            headers = {
                "Authorization": f"Bearer {YOUR_LONG_LIVED_ACCESS_TOKEN}",
                "Content-Type": "application/json"
            }

            # Send the request to Home Assistant
            try:
                response = requests.post(
                    f"{self.ha_url}/api/services/light/turn_on",
                    json=payload, headers=headers
                )
                response.raise_for_status()  # Raise an error for bad responses
                logger.info(f"Successfully set light color to XY: {xy_color} with brightness: {brightness}%")
                self.last_light_color = xy_color  # Update the last light color
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to set light color: {e}")

    def plex_monitor(self):
        """Monitor Plex for currently playing media and call the video-color function."""
        average_loop_time_milliseconds = 0
        while True:
            loop_start = time.time()  # Start time for the loop
            try:
                sessions = self.plex.sessions()
                if sessions:
                    for session in sessions:
                        video_path = self.get_video_path(session)

                        if video_path is None:
                            logger.error(f"Could not find media path for: {str(session)}")
                            continue

                        video_path = f"W:/{video_path[7:]}"

                        # Directly call the video color calculation function
                        avg_color = self.get_average_color(
                            video_path, session.viewOffset,
                            average_loop_time_milliseconds
                        )
                        self.log_average_color(avg_color)

                        # Set the light color based on the average color
                        if avg_color is not None:
                            self.set_light_color(avg_color, average_loop_time_milliseconds)

                else:
                    time.sleep(10)

            except Exception as e:
                logger.error(f"Error while monitoring Plex: {e}")
                time.sleep(30)  # In case of an error, wait 30 seconds before retrying

            # Calculate loop duration and log average time every 10 seconds
            average_loop_time_milliseconds = self.log_loop_duration(loop_start, average_loop_time_milliseconds)

    def get_video_path(self, session):
        """Get video path from session.guid."""
        video_path = None

        # Check session cache first for GUID to video path mapping
        with cache_lock:
            if session.guid in session_cache:
                video_path = session_cache[session.guid][0]
            else:
                try:
                    video_path = self.plex.library.section("films").getGuid(session.guid).media[0].parts[0].file
                    session_cache[session.guid] = (video_path, datetime.now())  # Cache the video path
                except Exception as e:
                    logger.warning(f"Failed to get video path from films section: {e}")

                if video_path is None:
                    try:
                        video_path = self.plex.library.section("tv programmes").getGuid(session.grandparentGuid).get(
                            session.title).media[0].parts[0].file
                        session_cache[session.guid] = (video_path, datetime.now())  # Cache the video path
                        logger.info(f"Found video path in TV programmes: {video_path}")
                    except Exception as e:
                        logger.warning(f"Failed to get video path from TV programmes section: {e}")

        return video_path

    def log_average_color(self, avg_color):
        """Log the average color if it has changed."""
        if avg_color is not None:
            # Log average color only if it has changed
            if self.last_avg_color is None or not np.array_equal(avg_color, self.last_avg_color):
                logger.info(f"Average Color: {avg_color}")
                self.last_avg_color = avg_color
        else:
            # Log the frame retrieval failure at defined intervals
            current_time = time.time()
            if current_time - self.last_frame_log_time >= FRAME_LOG_INTERVAL:
                logger.info(f"Could not retrieve frame from the video.")
                self.last_frame_log_time = current_time

    def log_loop_duration(self, loop_start, average_loop_time_milliseconds):
        """Calculate loop duration and log average time every 10 seconds."""
        loop_duration = time.time() - loop_start
        self.loop_times.append(loop_duration)

        if time.time() - self.loop_start_time >= 10:  # Log every 10 seconds
            average_loop_time = sum(self.loop_times) / len(self.loop_times)
            average_loop_time_milliseconds = average_loop_time * 1000
            logger.info(f"Average loop execution time: {average_loop_time_milliseconds :.3f} milliseconds")
            self.loop_times = []  # Reset for the next 10-second period
            self.loop_start_time = time.time()
        return average_loop_time_milliseconds


if __name__ == '__main__':
    plex_monitor_instance = PlexMonitor(
        PLEX_SERVER_URL,
        PLEX_TOKEN,
        HA_URL,  # Home Assistant URL
        HA_ENTITY_ID  # Entity ID of the lights
    )

    # Start the background cache cleanup process
    cleanup_process = multiprocessing.Process(target=plex_monitor_instance.background_cache_cleanup, daemon=True)
    cleanup_process.start()

    # Start the Plex monitoring thread
    plex_thread = multiprocessing.Process(target=plex_monitor_instance.plex_monitor, daemon=True)
    plex_thread.start()

    logger.info("Plex monitor started...")

    # Keep the main process alive
    while True:
        time.sleep(1)
