# Plex Server details
PLEX_SERVER_URL = "http://10.0.0.137:32400"  # Replace with actual Plex server URL

CACHE_EXPIRY_HOURS = 12
FRAME_LOG_INTERVAL = 10  # Log interval in seconds

# HA_URL = "http://homeassistant.local:8123/"
HA_URL = "http://10.0.0.214:8123/"

# HA_ENTITY_ID = "light.living_room_lights_2"
HA_LEFT_LIGHT = "light.hue_living_room_1"
HA_RIGHT_LIGHT = "light.hue_living_room_2"
# HA_ENTITY_ID = "light.bedroom_2_bulb"

TIME_TO_SET_LIGHT_MS = 1_500
AVERAGE_OVER_MS = 130
TV_PLAYER = 'BRAVIA VH2'
