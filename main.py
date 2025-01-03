import logging
import multiprocessing
import os
import time
from datetime import datetime, timedelta
from threading import Thread

import coloredlogs
import cv2
import numpy as np
import requests
from plexapi.server import PlexServer

from config import *
from secrets import PLEX_TOKEN, YOUR_LONG_LIVED_ACCESS_TOKEN

# Cache to store video file connections and last access time
video_cache = {}
session_cache = {}
cache_lock = multiprocessing.Lock()


class LightController:
    def __init__(self, ha_url, ha_entity_id):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='INFO', logger=self.logger)

        self.ha_url = ha_url
        self.ha_entity_id = ha_entity_id
        self.last_avg_color = None
        self.last_light_color = None
        self.last_light_update_time = 0

    @staticmethod
    def rgb_to_xy(r, g, b):
        """Convert RGB to CIE 1931 XY color space."""

        def gamma_correction(channel):
            return ((channel + 0.055) / 1.055) ** 2.4 if channel > 0.04045 else channel / 12.92

        r, g, b = [gamma_correction(channel / 255.0) for channel in (r, g, b)]
        X = r * 0.4124 + g * 0.3576 + b * 0.1805
        Y = r * 0.2126 + g * 0.7152 + b * 0.0722
        Z = r * 0.0193 + g * 0.1192 + b * 0.9505
        return (X / (X + Y + Z), Y / (X + Y + Z)) if X + Y + Z else (0, 0)

    def set_light_color(self, color, avg_loop_time_ms):
        """Set the light color via Home Assistant API."""
        current_time_ms = time.time() * 1000
        xy_color = self.rgb_to_xy(*color)
        brightness_pct = int(max(color) / 255.0 * 100)

        if self._should_update_light(xy_color):
            payload = {
                "entity_id": self.ha_entity_id,
                "xy_color": list(xy_color),
                "brightness_pct": brightness_pct,
                "transition": (avg_loop_time_ms / 1000) * 8,
            }
            self._send_light_update(payload)
            self.last_light_color = xy_color
            self.last_light_update_time = current_time_ms

    def _should_update_light(self, xy_color):
        """Determine if the light color needs to be updated."""
        return self.last_light_color is None or xy_color != self.last_light_color

    def _send_light_update(self, payload):
        """Send a request to update the light's color."""
        headers = {
            "Authorization": f"Bearer {YOUR_LONG_LIVED_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(f"{self.ha_url}/api/services/light/turn_on", json=payload, headers=headers)
            response.raise_for_status()
            self.logger.info(
                f"Set light {self.ha_entity_id} to color {payload['xy_color']} with brightness {payload['brightness_pct']}%"
            )
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to set light color: {e}")


class PlexMonitor:
    def __init__(self, plex_server_url, plex_token, ha_url):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        coloredlogs.install(level='INFO', logger=self.logger)

        self.plex = PlexServer(plex_server_url, plex_token)
        self.ha_url = ha_url
        self.left_light = LightController(ha_url, HA_LEFT_LIGHT)
        self.right_light = LightController(ha_url, HA_RIGHT_LIGHT)
        self.loop_times = []
        self.loop_start_time = time.time()
        # self.last_frame_log_time = 0
        # self.last_avg_color = None
        # self.last_light_color = None
        # self.last_light_update_time = 0

    def background_cache_cleanup(self):
        """Background process that removes old cache items every hour."""
        while True:
            time.sleep(3600)
            self.clean_old_cache_items()

    def clean_old_cache_items(self):
        """Remove video cache items that haven't been accessed in the last 12 hours."""
        current_time = datetime.now()

        with cache_lock:
            self._cleanup_video_cache(current_time)
            self._cleanup_session_cache(current_time)

    def _cleanup_video_cache(self, current_time):
        items_to_remove = [
            video_path for video_path, (_, last_access_time) in video_cache.items()
            if current_time - last_access_time > timedelta(hours=CACHE_EXPIRY_HOURS)
        ]

        for video_path in items_to_remove:
            self.logger.info(f"Removing cached video: {video_path}")
            video_cache[video_path][0].release()
            del video_cache[video_path]

    def _cleanup_session_cache(self, current_time):
        items_to_remove = [
            guid for guid in session_cache
            if current_time - session_cache[guid][1] > timedelta(hours=CACHE_EXPIRY_HOURS)
        ]

        for guid in items_to_remove:
            self.logger.info(f"Removing cached session GUID: {guid}")
            del session_cache[guid]

    def get_average_color(self, video_path, playback_time_ms, avg_loop_time_ms):
        """Retrieve average color for left and right halves."""
        cap = self._get_video_capture(video_path)
        if cap is None:
            return None, None

        frame_count = self._calculate_frame_count(cap, avg_loop_time_ms)
        cap.set(cv2.CAP_PROP_POS_MSEC, playback_time_ms + avg_loop_time_ms + TIME_TO_SET_LIGHT_MS)
        frames = self._retrieve_frames(cap, frame_count)
        if len(frames) == 0:
            return None, None

        left_frames = [frame[:, :frame.shape[1] // 2, :] for frame in frames]
        right_frames = [frame[:, frame.shape[1] // 2:, :] for frame in frames]

        left_color_avg = np.mean(np.stack(left_frames, axis=0), axis=(0, 1, 2))[::-1].tolist()
        right_color_avg = np.mean(np.stack(right_frames, axis=0), axis=(0, 1, 2))[::-1].tolist()

        return left_color_avg, right_color_avg

    def _get_video_capture(self, video_path):
        with cache_lock:
            if video_path not in video_cache:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return None
                video_cache[video_path] = (cap, datetime.now())
            else:
                cap, _ = video_cache[video_path]
                video_cache[video_path] = (cap, datetime.now())

        return cap

    def _calculate_frame_count(self, cap, avg_loop_time_ms):
        fps = cap.get(cv2.CAP_PROP_FPS)
        return int((AVERAGE_OVER_MS + avg_loop_time_ms) / 1000 * fps)

    def _retrieve_frames(self, cap, frame_count):
        frames = []
        for i in range(frame_count):
            if not cap.grab():
                self.logger.warning(f"Failed to grab frame {i + 1}/{frame_count}")
                continue

            ret, frame = cap.retrieve()
            if ret:
                frames.append(frame)
            else:
                self.logger.warning(f"Failed to retrieve frame {i + 1}/{frame_count}")

        return frames

    def plex_monitor(self):
        """Monitor Plex for currently playing media and call the video-color function."""
        avg_loop_time_ms = 0

        while True:
            loop_start = time.time()
            sessions = self._get_sessions()

            if sessions:
                for session in sessions:
                    if TV_PLAYER in session.player.title:
                        video_path = self.get_video_path(session)
                        if video_path:
                            if os.name == 'nt':
                                video_path = f"W:{video_path[7:]}"

                            left_color, right_color = self.get_average_color(
                                video_path, session.viewOffset, avg_loop_time_ms
                            )
                            if left_color and right_color:
                                self.left_light.set_light_color(left_color, avg_loop_time_ms)
                                self.right_light.set_light_color(right_color, avg_loop_time_ms)

            avg_loop_time_ms = self.log_loop_duration(loop_start, avg_loop_time_ms)

    def _get_sessions(self):
        try:
            return self.plex.sessions()
        except Exception as e:
            self.logger.error(f"Error while monitoring Plex: {e}")
            time.sleep(30)
            return []

    def get_video_path(self, session):
        """Get video path from session GUID."""
        with cache_lock:
            if session.guid in session_cache:
                return session_cache[session.guid][0]

            video_path = self._lookup_video_path(session)
            if video_path:
                session_cache[session.guid] = (video_path, datetime.now())

        return video_path

    def _lookup_video_path(self, session):
        try:
            return self.plex.library.section("films").getGuid(session.guid).media[0].parts[0].file
        except Exception:
            return self._lookup_tv_path(session)

    def _lookup_tv_path(self, session):
        try:
            return self.plex.library.section("tv programmes").getGuid(session.grandparentGuid).get(session.title).media[
                0].parts[0].file
        except Exception as e:
            self.logger.warning(f"Failed to get video path from TV programmes: {e}")
            return None

    def log_loop_duration(self, loop_start, avg_loop_time_ms):
        """Calculate loop duration and log average time every 10 seconds."""
        loop_duration = time.time() - loop_start
        self.loop_times.append(loop_duration)

        if time.time() - self.loop_start_time >= 10:
            avg_loop_time_ms = np.mean(self.loop_times) * 1000
            self.logger.info(f"Average loop time: {avg_loop_time_ms:.2f} ms")
            self.loop_start_time = time.time()
            self.loop_times.clear()

        return avg_loop_time_ms


if __name__ == '__main__':
    plex_monitor_instance = PlexMonitor(
        PLEX_SERVER_URL,
        PLEX_TOKEN,
        HA_URL,  # Home Assistant URL
    )

    # Start the background cache cleanup process
    # cleanup_process = multiprocessing.Process(target=plex_monitor_instance.background_cache_cleanup, daemon=True)
    cleanup_process = Thread(target=plex_monitor_instance.background_cache_cleanup, daemon=True)
    cleanup_process.start()

    # Start the Plex monitoring thread
    # plex_thread = multiprocessing.Process(target=plex_monitor_instance.plex_monitor, daemon=True)

    print("Plex monitor starting...")
    plex_monitor_instance.plex_monitor()
    # plex_thread = Thread(target=plex_monitor_instance.plex_monitor, daemon=True)
    # plex_thread.start()
    #
    #
    # # Keep the main process alive
    # while True:
    #     time.sleep(60)
