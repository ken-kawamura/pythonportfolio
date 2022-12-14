# -*- coding: utf-8 -*-
import cv2
import numpy as np

def red_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,127,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    return mask1 + mask2

# ブロブ解析
def analysis_blob(binary_img):
    # 2値画像のラベリング処理
    label = cv2.connectedComponentsWithStats(binary_img)

    # ブロブ情報を項目別に抽出
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)

    # ブロブ面積最大のインデックス
    max_index = np.argmax(data[:, 4])

    # 面積最大ブロブの情報格納用
    maxblob = {}

    # 面積最大ブロブの各種情報を取得
    maxblob["upper_left"] = (data[:, 0][max_index], data[:, 1][max_index]) # 左上座標
    maxblob["width"] = data[:, 2][max_index]  # 幅
    maxblob["height"] = data[:, 3][max_index]  # 高さ
    maxblob["area"] = data[:, 4][max_index]   # 面積
    maxblob["center"] = center[max_index]  # 中心座標

    return maxblob

def main():
    videofile_path = "practice.mp4"

    # カメラのキャプチャ
    cap = cv2.VideoCapture(videofile_path)


    #追加 保存用 大きさ指定
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter('capture_practice.mp4', cv2.VideoWriter_fourcc(*'mp4v'),25, (width, height))


    # ファイルからフレームを1枚ずつ取得して動画処理後に保存する
    x_list = []
    y_list = []
    data = []


    while(cap.isOpened()):
        # フレームを取得
        ret, frame = cap.read()

        # 赤色検出
        mask = red_detect(frame)

        # マスク画像をブロブ解析（面積最大のブロブ情報を取得）
        target = analysis_blob(mask)

        # 面積最大ブロブの中心座標を取得
        center_x = int(target["center"][0])
        center_y = int(target["center"][1])

        # フレームに面積最大ブロブの中心周囲を円で描く
        cv2.circle(frame, (center_x, center_y), 30, (0, 200, 0),
                   thickness=3, lineType=cv2.LINE_AA)

        data.append([center_x, center_y])

        # 座標をリストに追加
        x_list.append(center_x)
        y_list.append(center_y)


        # 軌跡を残しつつマーカを描画(軌跡を連続とするために新規frameには過去の座標分もforで描画している)
        for i in range(len(x_list)):
            frame = cv2.drawMarker(frame,
                                (int(x_list[i]), int(y_list[i])),
                                color=(255, 255, 255),
                                markerType=cv2.MARKER_CROSS,
                                markerSize=10,
                                thickness=1,
                                line_type=cv2.LINE_4)


        #追加 保存用
        writer.write(frame)



        # 結果表示
        cv2.imshow("Frame", frame)
        #cv2.imshow("Mask", mask)


        # qキーが押されたら途中終了
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # CSVファイルに保存
        np.savetxt('practice.csv', np.array(data), delimiter=",")


    cap.release()
    #追加 保存用
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()