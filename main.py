import os
from yolo_agent import video_detection_tool
from deepseek_agent import analysis_tool
from mail_agent import mail_tool

def main():
    # 1) Kullanıcıdan video dosyasının yolunu al
    video_path = input("Lütfen video dosyasının tam yolunu girin: ").strip()
    if not os.path.exists(video_path):
        print("Girilen dosya bulunamadı!")
        return

    # 2) YOLO tespiti
    with open(video_path, 'rb') as video_file:
        detection_result = video_detection_tool({"video": video_file})
        print(detection_result)

    # YOLO çıktılarının kaydedildiği klasör ve detections.txt
    detections_folder = "detections"
    detections_file = os.path.join(detections_folder, "detections.txt")

    if not os.path.exists(detections_file):
        print("Tespit dosyası bulunamadı! Lütfen önce video tespit işlemini kontrol edin.")
        return

    # 3) DeepSeek analiz (analysis.txt oluşturur)
    analysis_result = analysis_tool(detections_file)
    print(analysis_result)

    analysis_file = os.path.join(detections_folder, "analysis.txt")
    if not os.path.exists(analysis_file):
        print("analysis.txt bulunamadı, e-posta gönderilemedi.")
        return

    # 4) Mail gönderimi (analiz tamamlandıktan sonra)
    # Detections klasöründe 1.png dosyası var mı?
    image_path = os.path.join(detections_folder, "1.png")
    if not os.path.exists(image_path):
        print("1.png bulunamadı, bu nedenle e-posta gönderilmeyecek.")
        return

    # mail_tool bir LangChain tool olduğu için tek parametre olarak dict bekler
    mail_result = mail_tool({
    "input_data": {
        "analysis_file": analysis_file,
        "image_file": image_path
    }
})
    print(mail_result)


if __name__ == "__main__":
    main()
