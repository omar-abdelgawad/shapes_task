import cv2
from detect_shapes import detect_shapes

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
ORIGIN = (10, 30)
COLOR = (0, 255, 0)
THICKNESS = 3
# mobile phone camera rtsp
# rtsp_url = "rtsp://admin:admin@10.10.221.71:1935"  # Replace with your RTSP stream URL
# gst_str = f"rtspsrc location={rtsp_url} latency=0 buffer-mode=auto width=640 height=480 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
rtsp_url = "rtsp://192.168.1.100:8554/unicast"
gst_str = f"rtspsrc location={rtsp_url} latency=0 buffer-mode=auto ! decodebin ! videoconvert ! appsink"


def main() -> int:
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("can't open cam")
            break
        frame, score = detect_shapes(frame, conf=0.7)
        cv2.putText(frame, f"{score=}", ORIGIN, FONT, FONT_SCALE, COLOR, THICKNESS)

        # print(score)
        cv2.imshow("RTSP Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    exit(main())
