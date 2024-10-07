import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_average_color(video_path, playback_time):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Set the frame position to the desired playback time
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    cap.set(cv2.CAP_PROP_POS_MSEC, playback_time * 1000)  # Set to specific playback time in milliseconds

    # Read the frame at that specific time
    ret, frame = cap.read()

    if not ret:
        return None

    # Calculate the average color of the frame (in BGR format)
    avg_color_per_row = np.average(frame, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    # Convert BGR to RGB (for consistency)
    avg_color_rgb = avg_color[::-1]

    # Release the video capture object
    cap.release()

    return avg_color_rgb

@app.route('/video-color', methods=['POST'])
def video_color():
    data = request.get_json()

    # Extract media info and playback info from the request
    video_path = data.get('media_info')
    playback_time = data.get('playback_info')

    # Ensure both are provided
    if not video_path or playback_time is None:
        return jsonify({'error': 'Both media_info and playback_info are required.'}), 400

    # Get the average color of the frame at the specified time
    avg_color = get_average_color(video_path, playback_time)

    if avg_color is None:
        return jsonify({'error': 'Could not retrieve frame from the video.'}), 500

    return jsonify({'average_color': avg_color.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
