# 👥 Customer Segmentation Web App

เว็บแอปแบ่งกลุ่มลูกค้าด้วย **K-Means Clustering** สร้างด้วย Streamlit

## ฟีเจอร์
- ใช้ข้อมูลตัวอย่าง (ลูกค้าจำลอง 500 คน) หรืออัปโหลดไฟล์ CSV ของตัวเอง
- เลือกตัวแปรที่ใช้แบ่งกลุ่มและจำนวนกลุ่ม (K) ได้
- ปรับสเกลข้อมูลด้วย StandardScaler และประเมินผลด้วย Silhouette Score
- กราฟเชิงโต้ตอบ: PCA 2D, เลือกแกนเอง 2D, 3D, Pie, Radar
- Elbow Method และ Silhouette เพื่อช่วยเลือกค่า K
- สรุปลักษณะแต่ละกลุ่มอัตโนมัติ และดาวน์โหลดผลลัพธ์เป็น CSV

## โครงสร้างไฟล์
| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | หน้าเว็บ (Streamlit) |
| `segmentation.py` | ตรรกะข้อมูลตัวอย่างและ K-Means |
| `requirements.txt` | รายการไลบรารี |

## วิธีรัน
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

## รูปแบบ CSV ที่อัปโหลดได้
ต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์ เช่น Age, Annual_Income, Spending_Score
