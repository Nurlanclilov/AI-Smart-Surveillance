from ultralytics import YOLO
import cv2, os, pytesseract, re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model = YOLO("yolo11n.pt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "video.mp4")
CAPTURE_FOLDER = os.path.join(BASE_DIR, "captures")

SAFE_IDS = {1, 3, 7}
CONFIDENCE = 0.25
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
OCR_INTERVAL = 5

os.makedirs(CAPTURE_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video acila bilmedi!")
    exit()

FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS <= 0:
    FPS = 15

last_camera_time = "TIME_NOT_FOUND"
last_printed_time = ""
last_ocr_frame = 0
person_recorders = {}


def read_camera_datetime(frame):
    try:
        h, w = frame.shape[:2]

        roi = frame[0:int(h * .18), int(w * .55):w]
        if roi.size == 0:
            return ""

        roi = cv2.resize(
            roi, None, fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)

        _, binary = cv2.threshold(
            gray, 150, 255, cv2.THRESH_BINARY
        )

        texts = [
            pytesseract.image_to_string(gray, config="--psm 7"),
            pytesseract.image_to_string(binary, config="--psm 7")
        ]

        text = max(texts, key=len).strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("O", "0").replace("o", "0")
        text = text.replace("I", "1").replace("l", "1")

        pattern = (
            r"(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
            r"|\d{4}[./-]\d{1,2}[./-]\d{1,2})\s+"
            r"([01]?\d|2[0-3])[:.]([0-5]\d)"
            r"(?:[:.]([0-5]\d))?"
        )

        match = re.search(pattern, text)

        if not match:
            return ""

        date_text = re.sub(r"[./]", "-", match.group(1))
        parts = date_text.split("-")

        if len(parts) != 3:
            return ""

        if len(parts[0]) == 4:
            year, month, day = map(int, parts)
        else:
            day, month, year = map(int, parts)
            if year < 100:
                year += 2000

        hour = int(match.group(2))
        minute = int(match.group(3))
        second = int(match.group(4) or 0)

        dt = datetime(year, month, day, hour, minute, second)
        return dt.strftime("%d/%m/%Y %H:%M:%S")

    except Exception:
        return ""


def safe_filename(text):
    return re.sub(
        r"[^0-9A-Za-z_-]", "_",
        text or "TIME_NOT_FOUND"
    )


def create_person_video(person_id):
    filename = (
        f"person_{person_id}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.avi"
    )

    filepath = os.path.join(CAPTURE_FOLDER, filename)

    writer = cv2.VideoWriter(
        filepath,
        cv2.VideoWriter_fourcc(*"MJPG"),
        FPS,
        (VIDEO_WIDTH * 2, VIDEO_HEIGHT)
    )

    return (writer, filepath) if writer.isOpened() else (None, filepath)


def close_person_video(person_id):
    data = person_recorders.pop(person_id, None)

    if not data:
        return

    data["writer"].release()

    duration = (
        data["last_seen"] - data["start_frame"] + 1
    ) / FPS

    old_path = data["filepath"]

    if not os.path.exists(old_path):
        return

    base, ext = os.path.splitext(old_path)

    new_name = (
        f"{base}_"
        f"{safe_filename(data['camera_time'])}_"
        f"{duration:.1f}sec{ext}"
    )

    try:
        os.rename(old_path, new_name)
        print(f"Video saxlanildi: {new_name}")
    except Exception:
        print(f"Video saxlanildi: {old_path}")


frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    original_frame = frame.copy()

    results = model.track(
        original_frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=CONFIDENCE,
        imgsz=1280
    )

    people = []

    if results[0].boxes is not None:
        for box in results[0].boxes:

            if int(box.cls[0]) != 0 or box.id is None:
                continue

            confidence = float(box.conf[0])

            if confidence < CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_id = int(box.id[0])

            people.append(
                (x1, y1, x2, y2, confidence, person_id)
            )

    current_ids = {p[5] for p in people}

    # OCR
    if frame_number - last_ocr_frame >= OCR_INTERVAL:

        detected_time = read_camera_datetime(original_frame)
        last_ocr_frame = frame_number

        if detected_time:
            last_camera_time = detected_time

            if (
                person_recorders and
                detected_time != last_printed_time
            ):
                print("Kamera vaxti:", detected_time)
                last_printed_time = detected_time

    for person_id in list(person_recorders):

        if person_id not in current_ids:
            close_person_video(person_id)

    security_frame = original_frame.copy()

    yellow = (0, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)

    cv2.putText(
        security_frame,
        f"Insan sayi: {len(people)}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        .5, yellow, 1, cv2.LINE_AA
    )

    cv2.putText(
        security_frame,
        f"CAM TIME: {last_camera_time}",
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        .5, yellow, 1, cv2.LINE_AA
    )

    cv2.putText(
        security_frame,
        f"VIDEO RECORD: {len(person_recorders)}",
        (15, 71),
        cv2.FONT_HERSHEY_SIMPLEX,
        .5, yellow, 1, cv2.LINE_AA
    )

    if len(people) > 3:
        cv2.putText(
            security_frame,
            "NORMAL REJIM - COK INSAN",
            (15, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5, yellow, 1, cv2.LINE_AA
        )

    security_video = cv2.resize(
        security_frame,
        (VIDEO_WIDTH, VIDEO_HEIGHT)
    )

    for x1, y1, x2, y2, confidence, person_id in people:

        color = green if person_id in SAFE_IDS else red
        label = (
            f"SAFE ID: {person_id}"
            if person_id in SAFE_IDS
            else f"ALERT ID: {person_id}"
        )

        cv2.rectangle(
            security_frame,
            (x1, y1), (x2, y2),
            color, 2
        )

        cv2.putText(
            security_frame,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5, color, 1, cv2.LINE_AA
        )

        if person_id not in SAFE_IDS:
            cv2.putText(
                security_frame,
                "!!! ALERT !!!",
                (
                    x1,
                    min(
                        security_frame.shape[0] - 10,
                        y2 + 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5, red, 1, cv2.LINE_AA
            )

        padding = 80

        cx1 = max(0, x1 - padding)
        cy1 = max(0, y1 - padding)
        cx2 = min(original_frame.shape[1], x2 + padding)
        cy2 = min(original_frame.shape[0], y2 + padding)

        crop = original_frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        zoom = cv2.resize(
            crop,
            (VIDEO_WIDTH, VIDEO_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )

        collage = cv2.hconcat(
            [security_video, zoom]
        )

        if person_id not in person_recorders:

            writer, filepath = create_person_video(person_id)

            if writer is None:
                continue

            person_recorders[person_id] = {
                "writer": writer,
                "filepath": filepath,
                "start_frame": frame_number,
                "last_seen": frame_number,
                "camera_time": last_camera_time
            }

        else:

            person_recorders[person_id]["last_seen"] = frame_number

            if last_camera_time != "TIME_NOT_FOUND":
                person_recorders[person_id]["camera_time"] = last_camera_time

        person_recorders[person_id]["writer"].write(collage)

    cv2.imshow("SECURITY CAMERA", security_frame)

    if people:

        target = max(
            people,
            key=lambda p: (p[2] - p[0]) * (p[3] - p[1])
        )

        tx1, ty1, tx2, ty2, _, _ = target

        padding = 50

        zx1 = max(0, tx1 - padding)
        zy1 = max(0, ty1 - padding)
        zx2 = min(original_frame.shape[1], tx2 + padding)
        zy2 = min(original_frame.shape[0], ty2 + padding)

        zoom_display = original_frame[zy1:zy2, zx1:zx2]

        if zoom_display.size:
            zoom_display = cv2.resize(
                zoom_display,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            cv2.imshow("AUTO ZOOM", zoom_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for person_id in list(person_recorders):
    close_person_video(person_id)

cap.release()
cv2.destroyAllWindows()

print("Proqram tamamlandi.")
print(f"Videolar: {CAPTURE_FOLDER}")
