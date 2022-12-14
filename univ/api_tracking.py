import cv2
import time

# トラッカーの選択
tracker = cv2.TrackerMIL_create()

# キャプチャー
cap = cv2.VideoCapture('kaiseki3.mp4')
cap.set(cv2.CAP_PROP_FPS, 30)
print(cap.get(cv2.CAP_PROP_FPS))

# 画面表示を遅らせる
time.sleep(1)

ret, frame = cap.read()

roi = cv2.selectROI(frame, False)

ret = tracker.init(frame, roi)

# ファイルからフレームを1枚ずつ取得して動画処理後に保存する
x_list = []
y_list = []

while True:

    ret, frame = cap.read()

    success, roi = tracker.update(frame)

    (x,y,w,h) = tuple(map(int,roi))

    if success:
        p1 = (x, y)
        p2 = (x+w, y+h)
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
    else :
        cv2.putText(frame, "Tracking failed!!", (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)

    # 座標をリストに追加
    x_list.append(x+w/2)
    y_list.append(y+h/2)

    # 軌跡を残しつつマーカを描画(軌跡を連続とするために新規frameには過去の座標分もforで描画している)
    for i in range(len(x_list)):
        frame = cv2.drawMarker(frame,
                                (int(x_list[i]), int(y_list[i])),
                                color=(255, 255, 255),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=10,
                                thickness=1,
                                line_type=cv2.LINE_4)


    # 結果表示
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 :
        break

cap.release()
cv2.destroyAllWindows()