"""
Fase 2: Edge Video Streaming & Input Calibration Pipeline
Skrip modular berbasis OOP untuk membaca live video stream menggunakan
kamera smartphone via background thread, kalibrasi Scanning Zone (ROI),
visualisasi Cybernetic HUD, dan pencatatan log performa sesi.

Author: Senior AI Engineer
Date: 2026
"""

import os
import sys
import time
import threading
import csv
import urllib.request
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# Pustaka ekstraksi fitur dan loading model
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import img_as_ubyte

# ============================================================================
# CONFIGURATION
# ============================================================================
# Sumber Kamera Utama:
# - Set ke integer (misal: 1) untuk Virtual Webcam (DroidCam)
# - Set ke string (misal: 'http://192.168.1.50:8080/video') untuk IP Stream URL
CAMERA_SOURCE = "http://10.135.213.156:4747/video"

# Folder logs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


# ============================================================================
# 1. OPTIMIZED THREADED FRAME ACQUISITION (Threaded Video Reader)
# ============================================================================
class MJPEGStreamReader:
    """
    Pembaca stream MJPEG langsung via urllib — digunakan ketika OpenCV tidak
    memiliki backend FFMPEG aktif dan tidak bisa membuka URL HTTP secara langsung.
    Memparse JPEG boundary dari chunked HTTP response dan men-decode setiap frame.
    """
    def __init__(self, url, timeout=5):
        self.url = url
        self.timeout = timeout
        self.stream = None
        self.width = 640
        self.height = 480

    def open(self):
        """Membuka koneksi HTTP ke URL stream."""
        try:
            self.stream = urllib.request.urlopen(self.url, timeout=self.timeout)
            return True
        except Exception as e:
            print(f"[MJPEG] Gagal membuka stream: {e}")
            return False

    def read_frame(self):
        """
        Membaca satu frame JPEG dari MJPEG stream.
        Mengembalikan (True, frame_ndarray) jika berhasil, atau (False, None).
        """
        if self.stream is None:
            return False, None
        try:
            buf = b""
            # Baca sampai menemukan JPEG end marker (\xff\xd9)
            while True:
                chunk = self.stream.read(4096)
                if not chunk:
                    return False, None
                buf += chunk
                # Cari JPEG start (\xff\xd8) dan end (\xff\xd9)
                a = buf.find(b"\xff\xd8")
                b_end = buf.find(b"\xff\xd9")
                if a != -1 and b_end != -1 and a < b_end:
                    jpg_data = buf[a:b_end + 2]
                    buf = buf[b_end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.height, self.width = frame.shape[:2]
                        return True, frame
        except Exception:
            return False, None

    def release(self):
        """Menutup koneksi HTTP."""
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

class ThreadedVideoReader:
    """
    Kelas pembuat video reader terpisah (background thread) untuk membaca frame
    secara asinkron, mencegah lag buffer OpenCV, dan menjaga kelancaran FPS.
    Mendukung dua mode: OpenCV VideoCapture (untuk indeks/URL lokal) dan
    MJPEGStreamReader (untuk URL HTTP langsung tanpa FFMPEG).
    """
    def __init__(self, source, fallback_indices=[]):
        """
        Inisialisasi ThreadedVideoReader.
        
        Parameters:
        -----------
        source : int atau str
            Sumber tangkapan kamera utama.
        fallback_indices : list of int
            Daftar indeks kamera alternatif jika sumber utama gagal.
        """
        self.source = source
        self.fallback_indices = fallback_indices
        self.cap = None           # OpenCV VideoCapture (mode indeks/lokal)
        self.mjpeg = None         # MJPEGStreamReader (mode URL HTTP langsung)
        self.use_mjpeg = False    # Flag: True jika memakai MJPEGStreamReader
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # State Pelacakan
        self.actual_source = source
        self.width = 640
        self.height = 480
        
        self._initialize_stream()

    def _initialize_stream(self):
        """
        Membuka koneksi video stream. Untuk URL string:
        1. Coba cv2.VideoCapture terlebih dahulu (jika FFMPEG tersedia)
        2. Fallback ke MJPEGStreamReader via urllib jika OpenCV gagal
        """
        is_string = isinstance(self.source, str)
        
        print(f"[CAMERA] Menghubungkan ke sumber utama: {self.source} (Tipe: {'URL/String' if is_string else 'Indeks Kamera'})")
        
        if is_string:
            # Coba OpenCV dulu
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                print(f"[WARN] OpenCV (FFMPEG) gagal membuka URL. Mencoba MJPEG manual via urllib...")
                self.cap = None
                # Fallback ke MJPEGStreamReader
                self.mjpeg = MJPEGStreamReader(self.source)
                if self.mjpeg.open():
                    # Baca satu frame untuk validasi dan mendapatkan resolusi
                    ret, test_frame = self.mjpeg.read_frame()
                    if ret and test_frame is not None:
                        self.use_mjpeg = True
                        self.width = self.mjpeg.width
                        self.height = self.mjpeg.height
                        print(f"[SUCCESS] MJPEG stream berhasil dibuka! Resolusi: {self.width}x{self.height}")
                        # Simpan frame pertama ke buffer
                        with self.lock:
                            self.frame = test_frame
                            self.ret = True
                        return
                    else:
                        print("[WARN] MJPEG stream terbuka tapi gagal membaca frame. Mungkin bukan MJPEG?")
                        self.mjpeg.release()
                        self.mjpeg = None
                else:
                    self.mjpeg = None
                    
                raise ValueError(
                    f"[ERROR] Gagal membuka URL stream: {self.source}\n"
                    "  >> Pastikan HP dan PC berada di jaringan yang sama (hotspot/WiFi).\n"
                    "  >> Pastikan DroidCam OBS aktif dan streaming di HP."
                )
            else:
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if self.width <= 0 or self.height <= 0:
                    self.width, self.height = 640, 480
                print(f"[SUCCESS] OpenCV berhasil membuka URL stream! Resolusi: {self.width}x{self.height}")
        else:
            # Mode indeks kamera (integer)
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                print(f"[WARN] Gagal terhubung ke sumber utama: {self.source}")
                for fb_idx in self.fallback_indices:
                    print(f"[CAMERA] Mencoba fallback ke Indeks Kamera: {fb_idx}...")
                    self.cap = cv2.VideoCapture(fb_idx)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.cap.isOpened():
                        print(f"[SUCCESS] Berhasil terhubung ke fallback Indeks Kamera: {fb_idx}")
                        self.actual_source = fb_idx
                        break
                        
            if not self.cap.isOpened():
                raise ValueError(
                    f"[ERROR] Tidak ada kamera di indeks {self.source} yang dapat dibuka."
                )
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.width <= 0 or self.height <= 0:
                self.width, self.height = 640, 480

    def start(self):
        """
        Memulai background thread untuk penangkapan frame secara berkala.
        """
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, args=())
        self.thread.daemon = True
        self.thread.start()
        print("[CAMERA] Background thread pembacaan frame dimulai.")
        return self

    def _update_loop(self):
        """
        Loop internal yang berjalan asinkron untuk mengambil frame terbaru.
        Mendukung mode OpenCV (cap) dan mode MJPEG manual (mjpeg).
        """
        while self.running:
            if self.use_mjpeg and self.mjpeg is not None:
                ret, frame = self.mjpeg.read_frame()
            else:
                ret, frame = self.cap.read()
                
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
            # Istirahat sejenak untuk menghindari CPU overloading
            if not self.use_mjpeg:
                time.sleep(0.005)

    def read(self):
        """
        Membaca frame terbaru yang disimpan di RAM.
        
        Returns:
        --------
        ret : bool
            True jika frame berhasil diambil.
        frame : ndarray
            Frame gambar terbaru (atau None).
        """
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        """
        Menghentikan background thread dan melepaskan resource kamera.
        """
        print("[CAMERA] Menghentikan background thread...")
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.mjpeg is not None:
            self.mjpeg.release()
            print("[CAMERA] MJPEG HTTP stream berhasil ditutup.")
        elif self.cap is not None:
            self.cap.release()
            print("[CAMERA] Resource kamera fisik berhasil dilepaskan (cap.release()).")

# ============================================================================
# 2. CYBERNETIC HUD OVERLAY & LIVE STREAM PIPELINE
# ============================================================================
class LiveStreamPipeline:
    """
    Kelas utama pengelola alur eksekusi Edge Video Streaming,
    kalibrasi ROI (Scanning Zone), rendering HUD overlay, dan logging sesi.
    """
    def __init__(self, source):
        """
        Inisialisasi LiveStreamPipeline.
        """
        self.source = source
        self.reader = None
        
        # FPS Tracking variables
        self.fps_list = []
        self.avg_fps = 0.0
        self.start_session_time = None
        
        # Inisialisasi Model & Scaler Fase 1
        self.model_path = PROJECT_ROOT / "models" / "surface_classifier.pkl"
        self.scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
        self.model_loaded = False
        self.clf = None
        self.scaler = None
        self.predicted_frames_count = 0
        
        self._load_models()
        
        # Variabel ROI Dinamis untuk Kalibrasi
        self.roi_x = None
        self.roi_y = None
        self.roi_size = 250
        self.show_roi_patch = False
        
        # Advanced Telemetry Logging Setup
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.telemetry_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_path = LOGS_DIR / f"telemetry_{self.telemetry_timestamp}.csv"
        self.csv_file = open(self.telemetry_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Frame_ID", "Predicted_Class", "Confidence_Score", "Processing_Time_ms"])

    def _load_models(self):
        """
        Memuat model classifier dan scaler dari folder models/.
        """
        print("[MODEL] Memuat model dan scaler dari folder models/...")
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                self.clf = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.model_loaded = True
                print("[SUCCESS] Model classifier dan scaler berhasil dimuat!")
            except Exception as e:
                print(f"[ERROR] Gagal memuat model/scaler: {str(e)}")
        else:
            print(f"[WARN] File model/scaler tidak ditemukan di {self.model_path.parent.name}/.")
        
    def _calculate_roi_coords(self, width, height, box_size=250):
        """
        Menghitung koordinat pojok ROI (250x250) agar tepat berada di tengah-bawah frame.
        
        Returns:
        --------
        x1, y1, x2, y2 : int
            Koordinat bounding box Scanning Zone.
        """
        x1 = (width - box_size) // 2
        x2 = x1 + box_size
        
        # Posisikan di bawah dengan margin 25px dari batas bawah layar
        y2 = height - 25
        y1 = y2 - box_size
        
        return x1, y1, x2, y2

    def extract_hybrid_features(self, roi_img, d=1, R=1):
        """
        Ekstraksi fitur hibrida GLCM dan LBP pada potongan gambar ROI (50x50).
        """
        img_gray = img_as_ubyte(roi_img)
        
        # GLCM Jarak d=1, 4 arah
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img_gray, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        glcm_vector = np.array([contrast, homogeneity, energy, correlation])
        
        # LBP Radius R=1, n_points = 8
        n_points = 8 * R
        lbp_map = local_binary_pattern(img_gray, n_points, R, method='uniform')
        
        # Hitung histogram LBP
        lbp_hist, _ = np.histogram(
            lbp_map.ravel(), 
            bins=np.arange(0, n_points + 3),
            range=(0, n_points + 2)
        )
        
        # Normalisasi histogram
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-10)
        
        # Gabungkan
        feature_vector = np.hstack([glcm_vector, lbp_hist])
        return feature_vector

    def _draw_cybernetic_hud(self, frame, fps, width, height, x1, y1, x2, y2, prediction=None, confidence=0.0, camera_error=False):
        """
        Menggambar overlay visual grafis dengan gaya Cybernetic/Sci-Fi HUD monokromatik Hijau Neon/Merah.
        """
        # Palet Warna: Hijau Neon untuk Nominal/Standby, Merah untuk Anomalous
        color_neon_green = (0, 255, 50)
        color_red = (0, 0, 255)
        
        # Jika terjadi kamera error, gambar teks warning merah besar di tengah frame
        if camera_error:
            text = "CAMERA ERROR - RETRYING"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thick = 2
            text_size = cv2.getTextSize(text, font, scale, thick)[0]
            tx = (width - text_size[0]) // 2
            ty = (height + text_size[1]) // 2
            
            # Efek blinking/kedip warning
            if int(time.time() * 3) % 2 == 0:
                cv2.rectangle(frame, (tx - 15, ty - text_size[1] - 15), (tx + text_size[0] + 15, ty + 15), (0, 0, 0), -1)
                cv2.rectangle(frame, (tx - 15, ty - text_size[1] - 15), (tx + text_size[0] + 15, ty + 15), (0, 0, 255), 2)
                cv2.putText(frame, text, (tx, ty), font, scale, (0, 0, 255), thick, cv2.LINE_AA)
            return

        # Default HUD color
        hud_color = color_neon_green
        sys_status_str = "SYS_STATUS: CALIBRATED / READY"
        surface_text = ""
        
        if prediction == 0:
            hud_color = color_neon_green
            sys_status_str = "SYS_STATUS: NOMINAL (OK)"
            surface_text = f"SURFACE: NOMINAL | CONF: {confidence:.1f}%"
        elif prediction == 1:
            # Merah Terang (Flashing/Alert)
            # Untuk efek berkedip: ganti-ganti warna merah menyala dan merah redup berdasarkan waktu
            if int(time.time() * 5) % 2 == 0:
                hud_color = color_red
            else:
                hud_color = (0, 0, 150) # Merah redup
            sys_status_str = "STATUS: ANOMALOUS (ALERT)"
            surface_text = f"STATUS: ANOMALOUS (ROUGH) | CONF: {confidence:.1f}%"
            
        # 1. Gambar Bounding Box utama area pemindaian (ROI)
        cv2.rectangle(frame, (x1, y1), (x2, y2), hud_color, 1)
        
        # 2. Gambar Pojok Garis HUD Tebal (HUD-style corner brackets)
        bracket_len = 18
        bracket_thick = 3
        # Top-Left Bracket
        cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), hud_color, bracket_thick)
        cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), hud_color, bracket_thick)
        # Top-Right Bracket
        cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), hud_color, bracket_thick)
        cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), hud_color, bracket_thick)
        # Bottom-Left Bracket
        cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), hud_color, bracket_thick)
        cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), hud_color, bracket_thick)
        # Bottom-Right Bracket
        cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), hud_color, bracket_thick)
        cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), hud_color, bracket_thick)
        
        # Label teks di atas Scanning Zone
        cv2.putText(
            frame, 
            f"SCANNING ZONE [{x2-x1}x{y2-y1}]", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            hud_color, 
            1, 
            cv2.LINE_AA
        )
        
        # 3. Panel Status Transparan HUD (Kiri Atas)
        overlay = frame.copy()
        # Perluas tinggi panel ke 130 piksel untuk menampilkan baris keempat (hasil prediksi)
        panel_h = 130 if prediction is not None else 110
        cv2.rectangle(overlay, (15, 15), (320, panel_h), (0, 0, 0), -1) # Latar panel gelap
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) # Blending transparansi 40%
        
        # Teks informasi status system
        cv2.putText(frame, sys_status_str, (25, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"RESOLUTION: {width} x {height}", (25, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS       : {fps:.1f}", (25, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
        
        # Baris ke-4 jika ada prediksi aktif
        if prediction is not None:
            cv2.putText(frame, surface_text, (25, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1, cv2.LINE_AA)
        
        # Efek kedip-kedip cybernetic simulasi
        indicator_color = hud_color
        if int(time.time()) % 2 == 0:
            cv2.circle(frame, (295, 31), 4, indicator_color, -1)
            cv2.putText(frame, "LIVE SCAN", (230, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, indicator_color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (295, 31), 4, (0, 100, 0) if prediction != 1 else (0, 0, 100), -1)
            cv2.putText(frame, "WAIT_CONN", (230, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 150, 0) if prediction != 1 else (0, 0, 150), 1, cv2.LINE_AA)
                        
        # Garis grid HUD tipis di panel status
        cv2.line(frame, (20, 43), (315, 43), hud_color, 1)

    def _write_session_log(self, actual_source, width, height):
        """
        Menyimpan performa sesi stream video ke dalam file log baru di logs/ saat aplikasi ditutup.
        """
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = LOGS_DIR / f"camera_session_{timestamp}.txt"
        
        # Hitung rata-rata FPS
        avg_fps = sum(self.fps_list) / len(self.fps_list) if len(self.fps_list) > 0 else 0.0
        
        # Tutup file telemetry CSV
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            self.csv_file.close()
            print(f"[TELEMETRY] Log telemetry berhasil disimpan ke: {self.telemetry_path.resolve()}")
        
        # Catat detail sesi
        with open(log_filepath, "w") as f:
            f.write("="*65 + "\n")
            f.write("              CAMERA STREAM SESSION PERFORMANCE REPORT\n")
            f.write("="*65 + "\n")
            f.write(f"Waktu Sesi Selesai : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sumber Input Awal  : {self.source}\n")
            f.write(f"Sumber Input Aktif : {actual_source}\n")
            f.write(f"Resolusi Kamera    : {width} x {height} pixels\n")
            f.write(f"Rata-rata FPS      : {avg_fps:.2f} FPS\n")
            f.write(f"Pelepasan Resource : [OK] cap.release() dipanggil secara aman\n")
            f.write(f"Status Load Model  : {'[OK] Model & Scaler Berhasil Dimuat' if self.model_loaded else '[WARN] Gagal/Tidak Dimuat'}\n")
            f.write(f"Total Frame Prediksi: {self.predicted_frames_count} frame\n")
            f.write("="*65 + "\n")
            
        print(f"\n[SUKSES] Laporan sesi kamera disimpan ke: {log_filepath.resolve()}")

    def run(self, max_frames=None):
        """
        Memulai pipeline pemrosesan video streaming dan perulangan visualisasi frame.
        
        Parameters:
        -----------
        max_frames : int, optional
            Membatasi total frame yang dibaca (hanya untuk pengujian otomatis).
        """
        try:
            # 1. Inisialisasi threaded reader
            self.reader = ThreadedVideoReader(self.source)
            self.reader.start()
        except Exception as e:
            print(f"[ERROR] Gagal menginisialisasi kamera: {str(e)}")
            sys.exit(1)
            
        width = self.reader.width
        height = self.reader.height
        actual_source = self.reader.actual_source
        
        # Tentukan koordinat ROI dinamis (inisialisasi di tengah-bawah)
        if self.roi_x is None or self.roi_y is None:
            self.roi_x = (width - self.roi_size) // 2
            y2 = height - 25
            self.roi_y = y2 - self.roi_size
            
        print("\n" + "="*70)
        print("          LIVE CAMERA SURFACE SCANNER - RUNNING")
        print("          Tekan tombol 'q' untuk keluar dengan aman.")
        print("          KONTROL REAL-TIME CALIBRATION:")
        print("            - W/S: Pindahkan ROI Atas / Bawah (Y-axis)")
        print("            - A/D: Pindahkan ROI Kiri / Kanan (X-axis)")
        print("            - +/-: Perbesar / Perkecil ROI")
        print("            - V  : Toggle Live Patch ROI (50x50)")
        print("="*70 + "\n")
        
        self.start_session_time = time.time()
        prev_time = time.time()
        frame_idx = 0
        
        while True:
            t_loop_start = time.time()
            
            # Keluar jika sudah mencapai limit frame (untuk testing headless)
            if max_frames is not None and frame_idx >= max_frames:
                print(f"[TEST] Mencapai limit frame ({max_frames}). Keluar loop.")
                break
                
            # 2. Ambil frame dari threaded reader dengan Minimal Crash Protection
            frame = None
            ret = False
            camera_error = False
            
            try:
                ret, frame = self.reader.read()
                if not ret or frame is None:
                    camera_error = True
            except Exception:
                camera_error = True
                
            if camera_error:
                # Generate black frame as fallback to show error message on HUD
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # Sleep briefly to avoid aggressive loop spin during retries
                time.sleep(0.1)
                
            frame_idx += 1
            
            if not camera_error:
                # Mirroring agar pergerakan feed kamera terasa alami
                frame = cv2.flip(frame, 1)
            
            # Konstrain dimensi ROI agar tidak melebihi frame boundaries
            self.roi_size = max(50, min(self.roi_size, min(width, height)))
            self.roi_x = max(0, min(self.roi_x, width - self.roi_size))
            self.roi_y = max(0, min(self.roi_y, height - self.roi_size))
            
            x1 = self.roi_x
            y1 = self.roi_y
            x2 = x1 + self.roi_size
            y2 = y1 + self.roi_size
            
            # 3. Crop area Scanning Zone (ROI)
            roi = frame[y1:y2, x1:x2] if not camera_error else None
            
            prediction = None
            confidence = 0.0
            
            # ----------------------------------------------------------------
            # Suntikkan Ekstraksi Fitur Hybrid & Prediksi Model Fase 1
            # ----------------------------------------------------------------
            if self.model_loaded and roi is not None and not camera_error:
                try:
                    t_inf_start = time.time()
                    
                    # 1. Konversi ke Grayscale & Resize ke 50x50 piksel (sesuai dimensi dataset training Cropped50x)
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (50, 50), interpolation=cv2.INTER_AREA)
                    
                    # 2. Jalankan ekstraksi fitur Hybrid (GLCM d=1 & LBP R=1)
                    feat = self.extract_hybrid_features(roi_resized, d=1, R=1)
                    
                    # 3. Transformasi fitur menggunakan scaler
                    feat_scaled = self.scaler.transform(feat.reshape(1, -1))
                    
                    # 4. Prediksi label kelas
                    prediction = int(self.clf.predict(feat_scaled)[0])
                    
                    # Ambil confidence score (probability)
                    proba = self.clf.predict_proba(feat_scaled)[0]
                    confidence = float(proba[prediction]) * 100.0
                    
                    self.predicted_frames_count += 1
                    
                    t_inf_end = time.time()
                    proc_time_ms = (t_inf_end - t_inf_start) * 1000.0
                    
                    # Tulis hasil prediksi ke file telemetry CSV secara real-time
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    self.csv_writer.writerow([timestamp_str, frame_idx, prediction, f"{confidence:.2f}", f"{proc_time_ms:.2f}"])
                    self.csv_file.flush()
                except Exception:
                    pass
            
            # 4. Hitung FPS Instan
            t_loop_end = time.time()
            duration = t_loop_end - prev_time
            prev_time = t_loop_end
            
            fps = 1.0 / duration if duration > 0 else 30.0
            # Simpan data FPS untuk dirata-ratakan nanti di log
            self.fps_list.append(fps)
            
            # 5. Gambar visual HUD dinamis (Warna & Status berdasarkan prediksi)
            self._draw_cybernetic_hud(frame, fps, width, height, x1, y1, x2, y2, prediction, confidence, camera_error)
            
            # 6. Tampilkan jendela ROI Live Patch jika diaktifkan
            if self.show_roi_patch and roi is not None and not camera_error:
                try:
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_resized = cv2.resize(roi_gray, (50, 50), interpolation=cv2.INTER_AREA)
                    cv2.imshow("ROI Live Patch", roi_resized)
                except Exception:
                    pass
            
            # 7. Tampilkan ke Window GUI
            if max_frames is None:
                cv2.imshow("Cybernetic HUD Scanner (Fase 2)", frame)
                
                # Cek input keyboard untuk pergeseran dan pengubahan ROI dinamis
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Tombol 'q' ditekan. Keluar dari aplikasi...")
                    break
                elif key == ord('w') or key == ord('W'):
                    self.roi_y -= 10
                elif key == ord('s') or key == ord('S'):
                    self.roi_y += 10
                elif key == ord('a') or key == ord('A'):
                    self.roi_x -= 10
                elif key == ord('d') or key == ord('D'):
                    self.roi_x += 10
                elif key == ord('+') or key == ord('='):
                    self.roi_size += 10
                    # Proportional centering shift
                    self.roi_x -= 5
                    self.roi_y -= 5
                elif key == ord('-') or key == ord('_'):
                    self.roi_size -= 10
                    # Proportional centering shift
                    self.roi_x += 5
                    self.roi_y += 5
                elif key == ord('v') or key == ord('V'):
                    self.show_roi_patch = not self.show_roi_patch
                    if not self.show_roi_patch:
                        try:
                            cv2.destroyWindow("ROI Live Patch")
                        except Exception:
                            pass
            else:
                # Simulasi pembatasan kecepatan FPS di testing mode
                time.sleep(0.03)
                
        # 8. Hentikan camera stream, lepaskan resource dan ekspor log
        self.reader.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._write_session_log(actual_source, width, height)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fase 2: Edge Video Streaming & Input Calibration")
    parser.add_argument("--test", type=int, default=None, help="Jalankan sejumlah N frame dalam headless test mode")
    args = parser.parse_args()
    
    # Inisialisasi pipeline dengan kamera source ter-konfigurasi
    pipeline = LiveStreamPipeline(source=CAMERA_SOURCE)
    
    try:
        pipeline.run(max_frames=args.test)
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt terdeteksi. Menghentikan aplikasi...")
        if pipeline.reader is not None:
            pipeline.reader.stop()
        cv2.destroyAllWindows()
        pipeline._write_session_log(pipeline.reader.actual_source if pipeline.reader else CAMERA_SOURCE, 
                                     pipeline.reader.width if pipeline.reader else 640, 
                                     pipeline.reader.height if pipeline.reader else 480)
