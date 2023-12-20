import cv2
def extract_frames(video_path, output_folder):
    # Mở video để đọc
    video = cv2.VideoCapture(video_path)
    # Kiểm tra xem video có được mở thành công hay không
    if not video.isOpened():
        print("Không thể mở video.")
        return
    # Khởi tạo biến đếm số frame
    frame_count = 0
    # Đọc frame từ video và lưu thành ảnh
    while True:
        # Đọc frame từ video
        ret, frame = video.read()

        # Kiểm tra xem có còn frame hay không
        if not ret:
            break

        # Tạo đường dẫn lưu trữ cho frame
        output_path = f"{output_folder}/frame_{frame_count}.jpg"

        # Lưu frame thành ảnh

        cv2.imwrite(output_path, frame)

        # Tăng biến đếm số frame
        frame_count += 1

    # Đóng video
    video.release()
# Đường dẫn tới video
video_path = "path của video cần tách frame"
# Thư mục đầu ra để lưu các frame
output_folder = " path Folder lưu ảnh"
# Gọi hàm để tách frame
extract_frames(video_path, output_folder)

