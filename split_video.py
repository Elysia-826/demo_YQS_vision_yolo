import cv2
import os


def extract_frames(video_path, output_dir, fps_to_save=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算每隔多少帧保存一张图
    frame_interval = max(1, int(video_fps / fps_to_save))

    frame_count = 0
    saved_count = 0

    print(f"[START] 正在解构视频: {video_path}, 原始FPS: {video_fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img_name = f"match_frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, img_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[SUCCESS] 视频切帧完成！共白嫖出 {saved_count} 张黄金打标素材。")


# 🛠️ 填入你裁剪好的比赛视频路径和你想存放图片的文件夹
extract_frames("C:\\Users\\YQS\\Desktop\\new_data\\5月27日.mp4", "C:\\Users\\YQS\\Desktop\\new_data\\images")