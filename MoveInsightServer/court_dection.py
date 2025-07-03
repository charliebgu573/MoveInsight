import cv2
import numpy as np
from ultralytics import YOLO

video_path = "test data/1000012222.mp4"
output_video_path = "output/output_demo.mp4"
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
if not ret:
    print("视频读取失败")
    exit()
h, w = frame.shape[:2]

# ------- 1. 框球场四角 -------
show_frame = frame.copy()
points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(show_frame, (x, y), 6, (0,0,255), -1)
        cv2.imshow("Select 4 corners of court", show_frame)
cv2.imshow("Select 4 corners of court", show_frame)
cv2.setMouseCallback("Select 4 corners of court", mouse_callback)
while len(points) < 4:
    cv2.imshow("Select 4 corners of court", show_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyWindow("Select 4 corners of court")
court_poly = np.array(points, dtype=np.int32)
src_pts = np.float32(points)

# ------- 2. 画分界线（网线） -------
split_line = []
def split_line_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(split_line) < 2:
        split_line.append((x, y))
        cv2.circle(show_frame, (x, y), 6, (255,0,0), -1)
        cv2.imshow("Draw split line", show_frame)
show_frame2 = frame.copy()
cv2.polylines(show_frame2, [court_poly], isClosed=True, color=(0,0,255), thickness=2)
cv2.imshow("Draw split line", show_frame2)
cv2.setMouseCallback("Draw split line", split_line_callback)
while len(split_line) < 2:
    cv2.imshow("Draw split line", show_frame2)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyWindow("Draw split line")
p1, p2 = split_line

def point_in_poly(point, poly):
    return cv2.pointPolygonTest(poly, point, False) >= 0

def point_on_side(point, p1, p2):
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return (x2-x1)*(y-y1) - (y2-y1)*(x-x1)

# ------- 3. 标准俯视球场参数与映射 -------
court_w, court_h = 244, 536  # 单位：像素

dst_pts = np.float32([
    [0, 0],                 # 左上（对方底线左端）
    [court_w-1, 0],         # 右上（对方底线右端）
    [court_w-1, court_h-1], # 右下（己方底线右端）
    [0, court_h-1]          # 左下（己方底线左端）
])
H, _ = cv2.findHomography(src_pts, dst_pts)
def map_to_court(foot_xy, H):
    pts = np.array([[foot_xy]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, H)
    return int(np.clip(mapped[0][0][0],0,court_w-1)), int(np.clip(mapped[0][0][1],0,court_h-1))

def draw_court(court_img):
    lw = 2
    white = (255,255,255)
    court_img[:] = 0
    mx, my = court_w/6.1, court_h/13.4
    cv2.rectangle(court_img, (0,0), (court_w-1, court_h-1), white, lw)
    cv2.rectangle(court_img, (int(0.46*mx),0), (court_w-1-int(0.46*mx),court_h-1), white, lw)
    cv2.line(court_img, (0,0), (court_w-1,0), white, lw)
    cv2.line(court_img, (0,court_h-1), (court_w-1,court_h-1), white, lw)
    cv2.line(court_img, (0, int(court_h/2)), (court_w-1, int(court_h/2)), white, lw)
    net_y = court_h/2
    front_srv = int(net_y - 1.98*my)
    back_srv = int(net_y + 1.98*my)
    cv2.line(court_img, (0, front_srv), (court_w-1, front_srv), white, lw)
    cv2.line(court_img, (0, back_srv), (court_w-1, back_srv), white, lw)
    cv2.line(court_img, (0, int(0.76*my)), (court_w-1, int(0.76*my)), white, lw)
    cv2.line(court_img, (0, court_h-1-int(0.76*my)), (court_w-1, court_h-1-int(0.76*my)), white, lw)
    center_x = court_w // 2
    cv2.line(court_img, (center_x, 0), (center_x, court_h-1), white, lw)
    return court_img

# ------- 4. 平滑函数 -------
def smooth_point(traj, new_point, alpha=0.4):
    if not traj:
        return new_point
    last = traj[-1]
    return (int(last[0] * (1-alpha) + new_point[0] * alpha),
            int(last[1] * (1-alpha) + new_point[1] * alpha))

traj_p1, traj_p2 = [], []
dist_p1, dist_p2 = 0.0, 0.0
move_thresh = 2  # 超过2像素才计为移动

# 像素转米的比例
px_to_meter = 13.4 / court_h

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = frame.copy()
    cv2.polylines(annotated, [court_poly], isClosed=True, color=(0,0,255), thickness=2)
    cv2.line(annotated, p1, p2, (255,0,0), 2)

    lower_side = []  # P1候选
    upper_side = []  # P2候选

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                foot = ((x1 + x2) // 2, y2)
                if point_in_poly(foot, court_poly):
                    side = point_on_side(foot, p1, p2)
                    if side >= 0:
                        lower_side.append((y2, foot, (x1, y1, x2, y2)))
                    else:
                        upper_side.append((y2, foot, (x1, y1, x2, y2)))
    lower_side = sorted(lower_side, key=lambda tup: -tup[0])[:1]
    upper_side = sorted(upper_side, key=lambda tup: -tup[0])[:1]

    p1_foot = p2_foot = None
    if lower_side:
        _, foot, (x1, y1, x2, y2) = lower_side[0]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(annotated, foot, 6, (0,180,0), -1)
        cv2.putText(annotated, 'P1', (foot[0]-10, foot[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        p1_foot = foot
    if upper_side:
        _, foot, (x1, y1, x2, y2) = upper_side[0]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.circle(annotated, foot, 6, (0,0,255), -1)
        cv2.putText(annotated, 'P2', (foot[0]-10, foot[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        p2_foot = foot

    # -------- 轨迹可视化+平滑+阈值 ---------
    court_canvas = np.zeros((court_h, court_w, 3), dtype=np.uint8)
    court_canvas = draw_court(court_canvas)

    if p1_foot is not None:
        mapped = map_to_court(p1_foot, H)
        mapped = smooth_point(traj_p1, mapped, alpha=0.4)
        if traj_p1:
            dx = mapped[0] - traj_p1[-1][0]
            dy = mapped[1] - traj_p1[-1][1]
            delta = (dx**2 + dy**2)**0.5
            if delta > move_thresh:
                dist_p1 += delta
        traj_p1.append(mapped)
        if len(traj_p1) > 5:
            traj_p1.pop(0)
    if p2_foot is not None:
        mapped = map_to_court(p2_foot, H)
        mapped = smooth_point(traj_p2, mapped, alpha=0.4)
        if traj_p2:
            dx = mapped[0] - traj_p2[-1][0]
            dy = mapped[1] - traj_p2[-1][1]
            delta = (dx**2 + dy**2)**0.5
            if delta > move_thresh:
                dist_p2 += delta
        traj_p2.append(mapped)
        if len(traj_p2) > 5:
            traj_p2.pop(0)
    for i in range(1, len(traj_p1)):
        cv2.line(court_canvas, traj_p1[i-1], traj_p1[i], (0,255,0), 2)
    for pt in traj_p1:
        cv2.circle(court_canvas, pt, 8, (0,255,0), -1)
    for i in range(1, len(traj_p2)):
        cv2.line(court_canvas, traj_p2[i-1], traj_p2[i], (0,0,255), 2)
    for pt in traj_p2:
        cv2.circle(court_canvas, pt, 8, (0,0,255), -1)

    # --------- 缩小canvas ---------
    small_canvas = cv2.resize(court_canvas, (court_w // 2, court_h // 2), interpolation=cv2.INTER_AREA)
    sc_h, sc_w = small_canvas.shape[:2]

    # 两行字信息（米为单位）
    bar_h = 16 * 2 + 8  # 两行字高度+间隔
    bar = np.zeros((bar_h, sc_w, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thickness = 1

    p1_text = f"P1 dist: {dist_p1 * px_to_meter:.2f} m"
    p2_text = f"P2 dist: {dist_p2 * px_to_meter:.2f} m"
    color_p1 = (0, 255, 0)
    color_p2 = (0, 0, 255)

    cv2.putText(bar, p1_text, (8, 16), font, font_scale, color_p1, font_thickness)
    cv2.putText(bar, p2_text, (8, 16 * 2 + 4), font, font_scale, color_p2, font_thickness)

    full_canvas = np.vstack([small_canvas, bar])

    # --------- 粘贴到主图右上角 ---------
    offset_x, offset_y = 30, 30
    H_, W_ = annotated.shape[:2]
    fc_h, fc_w = full_canvas.shape[:2]
    if fc_w + offset_x < W_ and fc_h + offset_y < H_:
        annotated[offset_y:offset_y + fc_h, W_ - offset_x - fc_w:W_ - offset_x] = full_canvas

    writer.write(annotated)
    cv2.imshow("Court-ROI Person Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"输出视频已保存为 {output_video_path}")
