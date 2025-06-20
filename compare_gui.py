from deepface import DeepFace
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import uuid

def save_temp_image(image_array):
    """
    บันทึก numpy array ของภาพไปยังไฟล์ชั่วคราวและคืนค่าพาธของไฟล์นั้น
    """
    if image_array is None:
        print("Error: image_array เป็น None ใน save_temp_image")
        return None
    if not isinstance(image_array, np.ndarray):
        print(f"Error: image_array ไม่ใช่ numpy array. ชนิด: {type(image_array)}")
        return None
    if image_array.size == 0:
        print("Error: image_array ว่างเปล่าใน save_temp_image")
        return None

    path = f"temp_{uuid.uuid4().hex}.jpg"
    try:
        # Gradio 4.x sometimes gives RGB. Ensure correct color order for OpenCV.
        if len(image_array.shape) == 3 and image_array.shape[-1] == 3:
            # Check if it's RGB (common for Gradio output) and convert to BGR for OpenCV
            # A simple check: if the last channel values are very different from first two, it might be RGB.
            # However, safer to assume Gradio Image component provides RGB if type="numpy".
            # OpenCV's imwrite expects BGR.
            cv2.imwrite(path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        elif len(image_array.shape) == 2: # Grayscale image
            cv2.imwrite(path, image_array)
        else:
            print(f"คำเตือน: รูปร่างอาร์เรย์ภาพไม่คาดคิด: {image_array.shape}. พยายามบันทึกตามเดิม.")
            cv2.imwrite(path, image_array)

        if not os.path.exists(path):
            print(f"Error: ไม่สามารถสร้างไฟล์ชั่วคราวที่ {path}")
            return None
        test_read = cv2.imread(path)
        if test_read is None:
            print(f"Error: ไม่สามารถอ่านไฟล์ชั่วคราวที่บันทึกไว้ได้ที่ {path}. ไฟล์อาจเสียหรือว่างเปล่า.")
            os.remove(path)
            return None
        print(f"Temporary image saved successfully to {path} with shape {test_read.shape}")
        return path
    except Exception as e:
        print(f"Error ในการบันทึกภาพชั่วคราวไปที่ {path}: {e}")
        return None

def resize_image(img, target_height):
    """
    ปรับขนาดภาพให้มีความสูงตามที่กำหนด โดยรักษาสัดส่วนเดิมไว้
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == target_height:
        return img
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    interpolation = cv2.INTER_AREA if target_height < h else cv2.INTER_LINEAR
    resized_img = cv2.resize(img, (target_width, target_height), interpolation=interpolation)
    return resized_img


def analyze_and_display(img1_np, img2_np):
    """
    วิเคราะห์ใบหน้าสองภาพด้วย DeepFace และแสดงผลลัพธ์ผ่าน Gradio
    """
    if img1_np is None:
        return None, "Error: ไม่ได้ให้ภาพที่ 1", "", "", "", "", ""
    if img2_np is None:
        return None, "Error: ไม่ได้ให้ภาพที่ 2", "", "", "", "", ""

    img1_path = None
    img2_path = None

    try:
        img1_path = save_temp_image(img1_np)
        img2_path = save_temp_image(img2_np)

        if img1_path is None or img2_path is None:
            return None, "Error: ไม่สามารถประมวลผลภาพเพื่อวิเคราะห์ได้ (ไม่สามารถบันทึกไฟล์ชั่วคราว).", "", "", "", "", ""

        print(f"กำลังลอง DeepFace.verify ด้วย img1_path: {img1_path}, img2_path: {img2_path}")
        result = DeepFace.verify(img1_path, img2_path, model_name='Facenet', detector_backend='retinaface', enforce_detection=False)
        print("DeepFace.verify สำเร็จ.")

        print(f"กำลังลอง DeepFace.analyze สำหรับ img1_path: {img1_path}")
        analysis1 = DeepFace.analyze(img1_path, actions=["age", "gender"], detector_backend='retinaface', enforce_detection=False)
        analysis1_data = analysis1[0] if analysis1 else {'age': 'N/A', 'gender': {}, 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}, 'landmarks': {}}
        dominant_gender1_text = "N/A"
        if isinstance(analysis1_data.get('gender'), dict) and analysis1_data['gender']:
            dominant_gender1 = max(analysis1_data['gender'], key=analysis1_data['gender'].get)
            dominant_gender1_conf = analysis1_data['gender'][dominant_gender1]
            dominant_gender1_text = f"{dominant_gender1} ({dominant_gender1_conf:.2f}%)"
        else:
            dominant_gender1_text = str(analysis1_data.get('gender', 'N/A'))
        print(f"DeepFace.analyze (img1) สำเร็จ. ข้อมูล: {analysis1_data}")

        print(f"กำลังลอง DeepFace.analyze สำหรับ img2_path: {img2_path}")
        analysis2 = DeepFace.analyze(img2_path, actions=["age", "gender"], detector_backend='retinaface', enforce_detection=False)
        analysis2_data = analysis2[0] if analysis2 else {'age': 'N/A', 'gender': {}, 'region': {'x': 0, 'y': 0, 'w': 0, 'h': 0}, 'landmarks': {}}
        dominant_gender2_text = "N/A"
        if isinstance(analysis2_data.get('gender'), dict) and analysis2_data['gender']:
            dominant_gender2 = max(analysis2_data['gender'], key=analysis2_data['gender'].get)
            dominant_gender2_conf = analysis2_data['gender'][dominant_gender2]
            dominant_gender2_text = f"{dominant_gender2} ({dominant_gender2_conf:.2f}%)"
        else:
            dominant_gender2_text = str(analysis2_data.get('gender', 'N/A'))
        print(f"DeepFace.analyze (img2) สำเร็จ. ข้อมูล: {analysis2_data}")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return None, "Error: ไม่สามารถโหลดรูปภาพได้", "", "", "", "", ""

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_height = min(h1, h2)
        img1_resized = resize_image(img1, target_height)
        img2_resized = resize_image(img2, target_height)

        scale_ratio1 = target_height / h1
        scale_ratio2 = target_height / h2

        def adjust_region(analysis_dict, scale):
            region = analysis_dict.get('region')
            if region and region.get('w', 0) > 0 and region.get('h', 0) > 0:
                region['x'] = int(region['x'] * scale)
                region['y'] = int(region['y'] * scale)
                region['w'] = int(region['w'] * scale)
                region['h'] = int(region['h'] * scale)
                if 'landmarks' in analysis_dict and analysis_dict['landmarks']:
                    for point_name, coords in analysis_dict['landmarks'].items():
                        coords[0] = int(coords[0] * scale)
                        coords[1] = int(coords[1] * scale)
            return analysis_dict

        analysis1_data = adjust_region(analysis1_data, scale_ratio1)
        analysis2_data = adjust_region(analysis2_data, scale_ratio2)

        def draw_face_info(img, analysis_data, dominant_gender_text):
            region = analysis_data.get('region')
            if region and region.get('w', 0) > 0 and region.get('h', 0) > 0:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if 'landmarks' in analysis_data and analysis_data['landmarks']:
                    for point_name, coords in analysis_data['landmarks'].items():
                        px, py = int(coords[0]), int(coords[1])
                        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                            cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

                age_text = f"Age: {analysis_data.get('age', 'N/A')}"
                gender_text = f"Gender: {dominant_gender_text.split(' ')[0]}"
                y_age = max(15, y - 30)
                y_gender = max(35, y - 10)
                cv2.putText(img, age_text, (x, y_age), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, gender_text, (x, y_gender), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(img, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

        img1_drawn = draw_face_info(img1_resized.copy(), analysis1_data, dominant_gender1_text)
        img2_drawn = draw_face_info(img2_resized.copy(), analysis2_data, dominant_gender2_text)

        combined = np.hstack([img1_drawn, img2_drawn])

        # --- เพิ่มลายน้ำบนภาพผลลัพธ์ (ปรับขนาดตามภาพ, เพิ่มเงา) ---
        watermark_text = "The images were compared using AI by the Spider Team." # ข้อความกระชับ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        text_color = (0, 0, 200) # สีแดงเข้มขึ้น (BGR)
        shadow_color = (255, 255, 255) # สีขาวสำหรับเงา

        # คำนวณ font_scale แบบ dynamic:
        # ปรับขนาดตามความสูงของภาพรวม. ค่า 800.0 เป็นค่าที่ปรับเพื่อให้ได้ขนาดที่เหมาะสม
        # min(1.5, ...) เพื่อกำหนดขนาดสูงสุด, ไม่ให้ใหญ่เกินไปสำหรับภาพความละเอียดสูง
        font_scale = min(1.0, combined.shape[0] / 800.0)
        # กำหนดขนาดขั้นต่ำ ไม่ให้เล็กเกินไปจนอ่านไม่ออกสำหรับภาพความละเอียดต่ำ
        if font_scale < 0.5:
             font_scale = 0.5
        
        # คำนวณขนาดข้อความด้วย font_scale ที่ได้
        text_size = cv2.getTextSize(watermark_text, font, font_scale, font_thickness)[0]
        shadow_offset = 1 # กำหนด offset สำหรับเงา (ระยะห่างของเงาจากข้อความหลัก)

        # กำหนดตำแหน่งลายน้ำ (มุมล่างขวาของภาพรวม)
        # 10 คือระยะห่างจากขอบ
        text_x = combined.shape[1] - text_size[0] - 10
        text_y = combined.shape[0] - 10

        # วาดเงา (ข้อความสีขาวที่เลื่อนไปเล็กน้อย)
        cv2.putText(combined, watermark_text, (text_x + shadow_offset, text_y + shadow_offset), 
                    font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
        # วาดข้อความจริง (ข้อความสีแดงเข้ม) ทับลงบนเงา
        cv2.putText(combined, watermark_text, (text_x, text_y), 
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        combined_pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        similarity = round((1 - result["distance"]) * 100, 2)
        match_result = "Match ✅" if result["verified"] else "Not Match ❌"

        return (combined_pil, match_result, f"{similarity} %",
                analysis1_data.get("age", "N/A"), dominant_gender1_text,
                analysis2_data.get("age", "N/A"), dominant_gender2_text)

    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการวิเคราะห์: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}", "N/A", "N/A", "N/A", "N/A", "N/A"
    finally:
        if img1_path and os.path.exists(img1_path):
            os.remove(img1_path)
            print(f"ลบไฟล์ชั่วคราว: {img1_path}")
        if img2_path and os.path.exists(img2_path):
            os.remove(img2_path)
            print(f"ลบไฟล์ชั่วคราว: {img2_path}")

# --- กำหนดค่า Gradio Interface ---
# ข้อความด้านล่างสุดของหน้า Gradio
footer_html = "<div style='text-align: center; padding-top: 10px; color: gray; font-size: 0.9em;'>จัดทำโดย Spider Official Team</div>"

demo = gr.Interface(
    fn=analyze_and_display,
    inputs=[
        gr.Image(type="numpy", label="ภาพที่ 1"),
        gr.Image(type="numpy", label="ภาพที่ 2")
    ],
    outputs=[
        gr.Image(label="ผลลัพธ์ใบหน้า"), # เปลี่ยนชื่อ label เพื่อความกระชับ
        gr.Text(label="ผลลัพธ์การเปรียบเทียบ"),
        gr.Text(label="เปอร์เซ็นต์ความเหมือน"),
        gr.Text(label="อายุ (ภาพ 1)"),
        gr.Text(label="เพศ (ภาพ 1)"),
        gr.Text(label="อายุ (ภาพ 2)"),
        gr.Text(label="เพศ (ภาพ 2)")
    ],
    title="AI เปรียบเทียบใบหน้า พร้อม Age / Gender",
    description="อัปโหลดภาพสองภาพเพื่อเปรียบเทียบใบหน้า และวิเคราะห์อายุ เพศ", # ปรับ description
    article=footer_html, # ใช้สำหรับข้อความด้านล่างสุดของหน้า Gradio
    allow_flagging="never"
)

# --- รัน Gradio App ---
# คุณสามารถระบุ host และ port ได้หากต้องการรันบนเซิร์ฟเวอร์
demo.launch() 
# หรือ demo.launch(server_name="0.0.0.0", server_port=7860, share=False) สำหรับการใช้งานจริงบนเซิร์ฟเวอร์