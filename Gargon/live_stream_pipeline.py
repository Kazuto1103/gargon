"""
Surface Roughness Detector - Live Stream Pipeline (Fase 2)
Skrip berbasis OOP untuk menangani inisialisasi streaming video kamera,
pengukuran real-time FPS, frame skipping, preprocessing in-memory,
dan tampilan overlay HUD (Scanning Zone) menggunakan OpenCV.
"""

import sys
import time
import random
import cv2
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_FOLDER = PROJECT_ROOT / "data"
KASAR_FOLDER = DATA_FOLDER / "kasar"
HALUS_FOLDER = DATA_FOLDER / "halus"


class LiveSurfaceScanner:
    """
    Kelas utama untuk mengelola streaming kamera secara real-time,
    mengatur preprocessing in-memory, optimalisasi FPS, dan HUD overlay.
    """
    
    def __init__(self, camera_index=0, width=640, height=480, frame_skip=3, test_mode=False):
        """
        Inisialisasi objek LiveSurfaceScanner.
        
        Parameters:
        -----------
        camera_index : int
            Indeks perangkat kamera OpenCV.
        width : int
            Lebar frame video.
        height : int
            Tinggi frame video.
        frame_skip : int
            Skala skipping frame untuk optimasi kalkulasi berat.
        test_mode : bool
            Jika True, jalankan loop tanpa visualisasi GUI (untuk testing).
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.frame_skip = frame_skip
        self.test_mode = test_mode
        
        # State variables
        self.cap = None
        self.use_fallback = False
        self.frame_count = 0
        
        # FPS Tracking
        self.prev_time = 0
        self.fps = 0.0
        self.fps_accum = []
        
        # Fallback generator variables
        self.fallback_images = []
        self.current_fallback_idx = 0
        self.ticks_since_last_image = 0
        
        # Dimensi Scanning Zone (Bounding Box tengah)
        self.box_size = 256
        self.x1 = (self.width - self.box_size) // 2
        self.y1 = (self.height - self.box_size) // 2
        self.x2 = self.x1 + self.box_size
        self.y2 = self.y1 + self.box_size
        
    def _init_camera(self):
        """
        Inisialisasi OpenCV VideoCapture.
        Mengaktifkan mode fallback jika kamera tidak tersedia atau sibuk.
        """
        print(f"[CAMERA] Mencoba menginisialisasi kamera (Index: {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set resolusi kamera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Cek apakah kamera berhasil dibuka
        if not self.cap.isOpened():
            print("[WARN] Kamera fisik tidak terdeteksi atau sedang digunakan.")
            print("[WARN] Beralih ke Generator Video Sintetis Fallback...")
            self.use_fallback = True
            self._load_fallback_resources()
        else:
            # Pastikan resolusi yang diset didukung
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w > 0 and actual_h > 0:
                self.width = actual_w
                self.height = actual_h
                # Hitung ulang koordinat box untuk resolusi aktual
                self.x1 = (self.width - self.box_size) // 2
                self.y1 = (self.height - self.box_size) // 2
                self.x2 = self.x1 + self.box_size
                self.y2 = self.y1 + self.box_size
            print(f"[SUCCESS] Kamera fisik berhasil dibuka. Resolusi: {self.width}x{self.height}")
            
    def _load_fallback_resources(self):
        """
        Memuat gambar kasar/halus dari folder data untuk digunakan sebagai data video alternatif.
        """
        search_paths = [KASAR_FOLDER, HALUS_FOLDER, DATA_FOLDER]
        for folder in search_paths:
            if folder.exists():
                images = list(folder.glob("*.tif")) + \
                         list(folder.glob("*.tiff")) + \
                         list(folder.glob("*.png")) + \
                         list(folder.glob("*.jpg"))
                if images:
                    self.fallback_images.extend(images)
                    
        # Hapus duplikat path
        self.fallback_images = sorted(list(set(self.fallback_images)))
        
        if not self.fallback_images:
            print("[ERROR] Tidak menemukan gambar sampel di folder data untuk fallback generator.")
            print("[INFO] Membuat frame noise statis sebagai fallback.")
        else:
            print(f"[INFO] Fallback generator memuat {len(self.fallback_images)} gambar sampel untuk simulasi aliran video.")

    def _generate_fallback_frame(self):
        """
        Menghasilkan frame video sintetis di dalam RAM (in-memory).
        Mensimulasikan gerakan permukaan jalan dengan rotasi/geseran tekstur gambar kasar/halus.
        """
        # Jika tidak ada gambar sampel, hasilkan frame noise statis
        if not self.fallback_images:
            frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            return frame
            
        # Ganti gambar simulasi setiap 60 frame (sekitar 2 detik di 30fps)
        self.ticks_since_last_image += 1
        if self.ticks_since_last_image >= 90:
            self.ticks_since_last_image = 0
            self.current_fallback_idx = (self.current_fallback_idx + 1) % len(self.fallback_images)
            
        img_path = self.fallback_images[self.current_fallback_idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            # Fallback jika gambar gagal dibaca
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "Error loading image", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
            
        # Resize ke resolusi target
        frame = cv2.resize(img, (self.width, self.height))
        
        # Simulasikan getaran kamera & pergeseran permukaan (sliding scan)
        shift_x = int(10 * np.sin(self.frame_count * 0.1))
        shift_y = int(5 * np.cos(self.frame_count * 0.15))
        
        # Shift matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REFLECT)
        
        # Tambahkan sedikit random noise untuk mensimulasikan noise sensor kamera nyata
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame

    def _preprocess_frame(self, frame):
        """
        Melakukan preprocessing in-memory pada area pemindaian (Scanning Zone).
        Mengonversi ke grayscale dan melakukan downsampling/standardisasi ukuran.
        
        Returns:
        --------
        roi_processed : ndarray
            Area pemindaian (ROI) dalam grayscale yang telah di-resize (256x256).
        """
        # 1. Ambil Region of Interest (ROI) dari Scanning Zone
        roi = frame[self.y1:self.y2, self.x1:self.x2]
        
        # 2. Konversi ke Grayscale (in-memory, no disk writing)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Downsampling / standardisasi ukuran ke 256x256 (jika ukuran box berubah)
        if roi_gray.shape != (256, 256):
            roi_processed = cv2.resize(roi_gray, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            roi_processed = roi_gray
            
        return roi_processed

    def _draw_hud(self, frame, status="STANDBY", prediction="CALIBRATING..."):
        """
        Menggambar Graphical Overlay (HUD) di atas frame video.
        Menampilkan FPS aktif, Scanning Zone bounding box, dan status deteksi.
        """
        # 1. Gambar Bounding Box "Scanning Zone" di tengah layar
        # Warna Cyan (B=255, G=255, R=0) dengan ketebalan garis 2px
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 255, 0), 2)
        
        # Gambar pojok siku-siku (HUD-style corners) untuk mempercantik tampilan
        len_corner = 20
        # Top-Left
        cv2.line(frame, (self.x1, self.y1), (self.x1 + len_corner, self.y1), (255, 255, 0), 4)
        cv2.line(frame, (self.x1, self.y1), (self.x1, self.y1 + len_corner), (255, 255, 0), 4)
        # Top-Right
        cv2.line(frame, (self.x2, self.y1), (self.x2 - len_corner, self.y1), (255, 255, 0), 4)
        cv2.line(frame, (self.x2, self.y1), (self.x2, self.y1 + len_corner), (255, 255, 0), 4)
        # Bottom-Left
        cv2.line(frame, (self.x1, self.y2), (self.x1 + len_corner, self.y2), (255, 255, 0), 4)
        cv2.line(frame, (self.x1, self.y2), (self.x1, self.y2 - len_corner), (255, 255, 0), 4)
        # Bottom-Right
        cv2.line(frame, (self.x2, self.y2), (self.x2 - len_corner, self.y2), (255, 255, 0), 4)
        cv2.line(frame, (self.x2, self.y2), (self.x2, self.y2 - len_corner), (255, 255, 0), 4)

        # Label untuk Scanning Zone
        cv2.putText(frame, "SCANNING ZONE (256x256)", (self.x1, self.y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # 2. Tampilkan Real-time FPS Counter di pojok kanan atas
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.width - 140, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 3. Tampilkan Teks Placeholder Status & Hasil Klasifikasi (Cyan/Hijau Neon)
        # Latar belakang gelap transparan untuk meningkatkan keterbacaan teks
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, self.height - 110), (320, self.height - 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Tambahkan teks HUD di area kiri bawah
        cv2.putText(frame, "SURFACE SCANNER HUD", (30, self.height - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Status    : {status}", (30, self.height - 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA) # Hijau Neon
                    
        cv2.putText(frame, f"Kekasaran : {prediction}", (30, self.height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA) # Cyan

        # Indikator jika sedang berjalan dalam Mode Fallback
        if self.use_fallback:
            cv2.putText(frame, "MODE FALLBACK: SIMULASI ALIRAN VIDEO", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA) # Orange

    def run(self):
        """
        Siklus utama pemrosesan aliran video (Video Loop).
        Membaca frame, menghitung FPS, melakukan skip frame, preprocessing,
        menggambar overlay, dan menampilkan window visualisasi.
        """
        # Inisialisasi kamera
        self._init_camera()
        
        print("\n" + "="*60)
        print("MEMULAI SCREEN SCANNER VIDEO LOOP")
        print("Tekan tombol 'q' pada jendela video untuk keluar.")
        print("="*60 + "\n")
        
        self.prev_time = time.time()
        
        while True:
            # Hentikan loop dalam mode uji coba (headless test) setelah beberapa frame
            if self.test_mode and self.frame_count >= 15:
                print(f"[TEST] Uji coba loop berhasil dijalankan selama {self.frame_count} frame.")
                break
                
            t_start = time.time()
            self.frame_count += 1
            
            # 1. Tangkap frame
            if self.use_fallback:
                frame = self._generate_fallback_frame()
                # Batasi FPS simulasi agar tidak terlalu kencang (target ~30 FPS)
                time.sleep(0.03)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Gagal membaca frame dari kamera. Menutup loop...")
                    break
                    
                # Mirroring gambar untuk kamera depan (agar pergerakan natural)
                frame = cv2.flip(frame, 1)

            # 2. Frame Skipping Logic & Preprocessing
            # Preprocessing area pemindaian hanya dilakukan setiap N frame (frame_skip)
            is_processed_this_frame = False
            if self.frame_count % self.frame_skip == 0:
                is_processed_this_frame = True
                # Jalankan preprocessing in-memory
                roi_gray = self._preprocess_frame(frame)
                
                # [Untuk Fase 3]: Di sini tempat pemanggilan inference model
                # prediction = model.predict(roi_gray)
                pass

            # 3. Hitung FPS nyata (Real-time FPS)
            t_end = time.time()
            frame_duration = t_end - self.prev_time
            self.prev_time = t_end
            
            if frame_duration > 0:
                instant_fps = 1.0 / frame_duration
                # Moving average FPS agar transisinya halus di layar
                self.fps_accum.append(instant_fps)
                if len(self.fps_accum) > 30:
                    self.fps_accum.pop(0)
                self.fps = sum(self.fps_accum) / len(self.fps_accum)

            # 4. Gambar Graphical Overlay (HUD)
            status_text = "SCANNING" if is_processed_this_frame else "STANDBY"
            self._draw_hud(frame, status=status_text, prediction="CALIBRATING...")

            # 5. Tampilkan frame ke jendela Windows
            if not self.test_mode:
                cv2.imshow("Surface Roughness Scanner (Fase 2)", frame)

            # 6. Event handling: keluar jika tombol 'q' ditekan
            if not self.test_mode:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Tombol 'q' ditekan. Menghentikan video stream...")
                    break
            else:
                # Tambahkan sedikit delay untuk mensimulasikan FPS di test mode
                time.sleep(0.01)
                
        # Tutup resource
        self._cleanup()

    def _cleanup(self):
        """
        Membersihkan dan melepaskan semua resource video stream dan menutup window.
        """
        print("[CLEANUP] Menutup koneksi kamera dan jendela visualisasi...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[SUCCESS] Cleanup selesai dengan sukses. Sampai jumpa!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Surface Roughness Scanner")
    parser.add_argument("--test", action="store_true", help="Jalankan dalam mode uji coba (headless, 15 frame)")
    args = parser.parse_args()

    # Menggunakan kamera indeks 0, resolusi standar 640x480, skip=3 frame
    scanner = LiveSurfaceScanner(camera_index=0, width=640, height=480, frame_skip=3, test_mode=args.test)
    try:
        scanner.run()
    except KeyboardInterrupt:
        print("\n[INFO] Deteksi KeyboardInterrupt. Menutup aplikasi...")
        scanner._cleanup()
