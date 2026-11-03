# src/gui.py
# =========================================================================
# MODULE GIAO DIỆN TKINTER - CỬA SỔ ĐIỀU KHIỂN CHÍNH
# - Chọn ảnh từ máy tính
# - Nhận diện biển số
# - Chọn vùng thủ công
# - Tạo nhiễu, giảm nhiễu
# - So sánh, lưu kết quả, xuất CSV
# - Xử lý OCR nặng ở worker thread, trả kết quả về luồng GUI
# =========================================================================

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import OrderedDict

import cv2
import numpy as np

import config
from src.image_io import load_image, save_image
from src.pipeline import process_image, process_plate_region
from src.noise import apply_noise
from src.denoise import apply_denoise
from src.visualization import draw_results, show_result_nonblocking, create_comparison_image
from src.ocr_engine import OCREngine
from src.evaluation import export_csv, format_result_for_csv, print_summary


class MainApp:
    """
    Cửa sổ điều khiển chính (tkinter).
    Tách biệt với cửa sổ ảnh kết quả (OpenCV).
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện biển số xe - Bảng điều khiển")
        screen_h = self.root.winfo_screenheight()
        win_w = 500
        win_h = min(600, max(480, screen_h - 110))
        self.root.geometry(f"{win_w}x{win_h}+15+15")
        self.root.minsize(450, 420)

        # ===================== TRẠNG THÁI =====================
        self.image_path = None        # Đường dẫn ảnh hiện tại
        self.original_image = None    # Ảnh gốc BGR (sau chuẩn hóa, trước nhiễu)
        self.current_image = None     # Ảnh đang xử lý (có thể đã có nhiễu)
        self.noisy_image = None       # Ảnh nhiễu
        self.denoised_image = None    # Ảnh sau giảm nhiễu
        self.scale_factor = 1.0       # Hệ số resize
        self.result = None            # Kết quả pipeline gốc
        self.noisy_result = None      # Kết quả pipeline nhiễu
        self.denoised_result = None   # Kết quả pipeline giảm nhiễu
        self.csv_rows = []            # Dữ liệu CSV tích lũy

        # OCR engine - khởi tạo một lần
        self.ocr_engine = OCREngine()
        self._ocr_initialized = False

        # Vùng biển thủ công
        self.manual_bbox = None

        # Chế độ ảnh: "auto" = ảnh toàn xe, "full_image" = ảnh biển đã cắt
        self.image_mode_var = tk.StringVar(value="auto")

        # Trạng thái xử lý
        self._processing = False

        # Cửa sổ OpenCV đang mở
        self._cv_windows = []

        # Thư mục nóng (Hot Folder)
        self._hot_folder_running = False
        self._hot_folder_thread = None
        self._hot_folder_seen = set()

        # ===================== TẠO GIAO DIỆN =====================
        self._build_gui()

        # Xử lý đóng cửa sổ và phím tắt
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key_press)

        # Cập nhật OpenCV windows định kỳ
        self._poll_cv()

    # =================================================================
    # PHẦN 1: XÂY DỰNG GIAO DIỆN
    # =================================================================

    def _build_gui(self):
        """Tạo giao diện điều khiển tkinter."""

        # Canvas cho scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical",
                                  command=main_canvas.yview)
        self.scroll_frame = ttk.Frame(main_canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        self._canvas_win_id = main_canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )
        main_canvas.bind(
            "<Configure>",
            lambda e: main_canvas.itemconfig(self._canvas_win_id, width=e.width)
        )
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Cho phép cuộn bằng chuột
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)

        parent = self.scroll_frame

        # --- NHÓM 1: CHỌN ẢNH ---
        frame_img = ttk.LabelFrame(parent, text="📷 Chọn ảnh", padding=5)
        frame_img.pack(fill="x", padx=6, pady=2)

        self.lbl_path = ttk.Label(frame_img, text="Chưa chọn ảnh", wraplength=440)
        self.lbl_path.pack(fill="x")

        btn_frame = ttk.Frame(frame_img)
        btn_frame.pack(fill="x", pady=2)

        self.btn_choose = ttk.Button(btn_frame, text="Chọn ảnh...",
                                     command=self._choose_image)
        self.btn_choose.pack(side="left", padx=2)

        self.btn_detect = ttk.Button(btn_frame, text="Nhận diện",
                                     command=self._run_detection, state="disabled")
        self.btn_detect.pack(side="left", padx=2)

        self.btn_restore = ttk.Button(btn_frame, text="Khôi phục ảnh gốc",
                                      command=self._restore_original, state="disabled")
        self.btn_restore.pack(side="left", padx=2)

        btn_hot_frame = ttk.Frame(frame_img)
        btn_hot_frame.pack(fill="x", pady=(1, 2))

        self.btn_hot_folder = ttk.Button(btn_hot_frame, text="📂 Thư mục nóng (test_live/)",
                                         command=self._toggle_hot_folder)
        self.btn_hot_folder.pack(side="left", padx=2)

        self.btn_open_folder = ttk.Button(btn_hot_frame, text="Mở thư mục",
                                          command=self._open_hot_folder)
        self.btn_open_folder.pack(side="left", padx=2)

        # --- Chế độ ảnh đầu vào ---
        mode_frame = ttk.Frame(frame_img)
        mode_frame.pack(fill="x", pady=(2, 0))

        ttk.Label(mode_frame, text="Loại ảnh:").pack(side="left")
        self.rb_auto = ttk.Radiobutton(
            mode_frame, text="🚗 Ảnh toàn xe (Tự động)",
            variable=self.image_mode_var, value="auto", command=self._on_input_mode_changed)
        self.rb_auto.pack(side="left", padx=4)

        self.rb_full = ttk.Radiobutton(
            mode_frame, text="🔲 Ảnh biển đã cắt",
            variable=self.image_mode_var, value="full_image", command=self._on_input_mode_changed)
        self.rb_full.pack(side="left", padx=4)

        self.lbl_hot_hint = ttk.Label(
            frame_img,
            text="💡 Thư mục nóng: Bật nút trên, sau đó thả ảnh bất kỳ vào thư mục 'test_live/' để tự động nhận diện tức thì.",
            foreground="#0b5ed7", wraplength=440)
        self.lbl_hot_hint.pack(fill="x", pady=(2, 0))

        # --- NHÓM 2: CHỌN VÙNG THỦ CÔNG ---
        frame_manual = ttk.LabelFrame(parent, text="✋ Chọn vùng biển thủ công",
                                       padding=5)
        frame_manual.pack(fill="x", padx=6, pady=2)

        coords_frame = ttk.Frame(frame_manual)
        coords_frame.pack(fill="x")

        ttk.Label(coords_frame, text="X:").pack(side="left")
        self.entry_x = ttk.Entry(coords_frame, width=5)
        self.entry_x.pack(side="left", padx=2)
        self.entry_x.insert(0, "0")

        ttk.Label(coords_frame, text="Y:").pack(side="left")
        self.entry_y = ttk.Entry(coords_frame, width=5)
        self.entry_y.pack(side="left", padx=2)
        self.entry_y.insert(0, "0")

        ttk.Label(coords_frame, text="W:").pack(side="left")
        self.entry_w = ttk.Entry(coords_frame, width=5)
        self.entry_w.pack(side="left", padx=2)
        self.entry_w.insert(0, "200")

        ttk.Label(coords_frame, text="H:").pack(side="left")
        self.entry_h = ttk.Entry(coords_frame, width=5)
        self.entry_h.pack(side="left", padx=2)
        self.entry_h.insert(0, "60")

        btn_manual_frame = ttk.Frame(frame_manual)
        btn_manual_frame.pack(fill="x", pady=2)

        self.btn_manual_select = ttk.Button(
            btn_manual_frame, text="Kéo chọn trên ảnh",
            command=self._select_roi_mouse, state="disabled")
        self.btn_manual_select.pack(side="left", padx=2)

        self.btn_manual_detect = ttk.Button(
            btn_manual_frame, text="Nhận diện vùng thủ công",
            command=self._run_manual_detection, state="disabled")
        self.btn_manual_detect.pack(side="left", padx=2)

        # --- NHÓM 3: TẠO NHIỄU ---
        frame_noise = ttk.LabelFrame(parent, text="🔊 Tạo nhiễu", padding=5)
        frame_noise.pack(fill="x", padx=6, pady=2)

        # Loại nhiễu
        noise_type_frame = ttk.Frame(frame_noise)
        noise_type_frame.pack(fill="x")

        ttk.Label(noise_type_frame, text="Loại nhiễu:").pack(side="left")
        self.noise_type_var = tk.StringVar(value="gaussian")
        noise_combo = ttk.Combobox(noise_type_frame, textvariable=self.noise_type_var,
                                   values=["gaussian", "salt_pepper", "blur", "brightness"],
                                   state="readonly", width=13)
        noise_combo.pack(side="left", padx=4)
        noise_combo.bind("<<ComboboxSelected>>", self._on_noise_type_changed)

        # Phạm vi nhiễu
        ttk.Label(noise_type_frame, text="Phạm vi:").pack(side="left", padx=(6, 0))
        self.noise_scope_var = tk.StringVar(value="full")
        scope_combo = ttk.Combobox(noise_type_frame, textvariable=self.noise_scope_var,
                                   values=["full", "roi"],
                                   state="readonly", width=7)
        scope_combo.pack(side="left", padx=4)

        # Thanh trượt thông số
        self.noise_param_frame = ttk.Frame(frame_noise)
        self.noise_param_frame.pack(fill="x", pady=2)

        ttk.Label(self.noise_param_frame, text="Mức:").pack(side="left")
        self.noise_scale = tk.Scale(
            self.noise_param_frame, from_=0, to=100,
            orient="horizontal", length=160)
        self.noise_scale.set(25)
        self.noise_scale.pack(side="left", padx=4, fill="x", expand=True)

        self.lbl_noise_hint = ttk.Label(self.noise_param_frame,
                                        text="(sigma=25.0)")
        self.lbl_noise_hint.pack(side="left")

        # Cập nhật hint khi trượt
        self.noise_scale.configure(command=self._update_noise_hint)

        # Seed + Nút tạo nhiễu trên cùng 1 hàng gọn gàng
        seed_frame = ttk.Frame(frame_noise)
        seed_frame.pack(fill="x", pady=2)

        ttk.Label(seed_frame, text="Seed:").pack(side="left")
        self.entry_seed = ttk.Entry(seed_frame, width=7)
        self.entry_seed.pack(side="left", padx=4)
        self.entry_seed.insert(0, "42")

        self.btn_add_noise = ttk.Button(
            seed_frame, text="Tạo nhiễu",
            command=self._add_noise, state="disabled")
        self.btn_add_noise.pack(side="left", padx=6)

        # Banner phím tắt Live Demo
        lbl_hotkeys = ttk.Label(
            frame_noise,
            text="⌨️ Phím tắt: [B] Mờ | [D] Tối | [N] Nhiễu hạt | [S] Muối tiêu | [G] Giảm nhiễu | [R] Gốc",
            foreground="#b02a37", font=("Segoe UI", 8, "bold"), wraplength=440)
        lbl_hotkeys.pack(fill="x", pady=(2, 0))

        # --- NHÓM 4: GIẢM NHIỄU ---
        frame_denoise = ttk.LabelFrame(parent, text="🔇 Giảm nhiễu", padding=5)
        frame_denoise.pack(fill="x", padx=6, pady=2)

        denoise_type_frame = ttk.Frame(frame_denoise)
        denoise_type_frame.pack(fill="x")

        ttk.Label(denoise_type_frame, text="Bộ lọc:").pack(side="left")
        self.denoise_type_var = tk.StringVar(value="median")
        denoise_combo = ttk.Combobox(
            denoise_type_frame, textvariable=self.denoise_type_var,
            values=["median", "gaussian", "bilateral", "nlm"],
            state="readonly", width=13)
        denoise_combo.pack(side="left", padx=4)

        self.btn_denoise = ttk.Button(
            denoise_type_frame, text="Giảm nhiễu & Nhận diện lại",
            command=self._run_denoise, state="disabled")
        self.btn_denoise.pack(side="left", padx=6)

        # Thanh trượt thông số giảm nhiễu
        self.denoise_param_frame = ttk.Frame(frame_denoise)
        self.denoise_param_frame.pack(fill="x", pady=2)

        ttk.Label(self.denoise_param_frame, text="Mức:").pack(side="left")
        self.denoise_scale = tk.Scale(
            self.denoise_param_frame, from_=3, to=31,
            orient="horizontal", length=160, resolution=2)
        self.denoise_scale.set(5)
        self.denoise_scale.pack(side="left", padx=4, fill="x", expand=True)

        # Gợi ý bộ lọc
        self.lbl_denoise_hint = ttk.Label(
            frame_denoise,
            text="💡 Gợi ý: Median cho muối tiêu, Gaussian/NLM cho nhiễu Gaussian",
            wraplength=440, foreground="gray")
        self.lbl_denoise_hint.pack(fill="x")

        # --- NHÓM 5: SO SÁNH VÀ LƯU ---
        frame_actions = ttk.LabelFrame(parent, text="📊 So sánh và lưu kết quả",
                                        padding=5)
        frame_actions.pack(fill="x", padx=6, pady=2)

        btn_actions_frame = ttk.Frame(frame_actions)
        btn_actions_frame.pack(fill="x")

        self.btn_compare = ttk.Button(
            btn_actions_frame, text="Xem so sánh",
            command=self._show_comparison, state="disabled")
        self.btn_compare.pack(side="left", padx=2)

        self.btn_save = ttk.Button(
            btn_actions_frame, text="Lưu ảnh kết quả",
            command=self._save_results, state="disabled")
        self.btn_save.pack(side="left", padx=2)

        self.btn_csv = ttk.Button(
            btn_actions_frame, text="Xuất CSV",
            command=self._export_csv, state="disabled")
        self.btn_csv.pack(side="left", padx=2)

        # --- NHÓM 6: TRẠNG THÁI VÀ KẾT QUẢ ---
        frame_status = ttk.LabelFrame(parent, text="📋 Trạng thái và kết quả",
                                       padding=5)
        frame_status.pack(fill="both", padx=6, pady=2, expand=True)

        self.text_status = tk.Text(frame_status, height=8, wrap="word",
                                   state="disabled", font=("Consolas", 11, "bold"))
        status_scroll = ttk.Scrollbar(frame_status, command=self.text_status.yview)
        self.text_status.configure(yscrollcommand=status_scroll.set)

        self.text_status.pack(side="left", fill="both", expand=True)
        status_scroll.pack(side="right", fill="y")

    # =================================================================
    # PHẦN 2: LOGIC XỬ LÝ
    # =================================================================

    def _log(self, msg):
        """Ghi thông báo vào vùng trạng thái."""
        self.text_status.configure(state="normal")
        self.text_status.insert("end", msg + "\n")
        self.text_status.see("end")
        self.text_status.configure(state="disabled")

    def _clear_log(self):
        """Xóa vùng trạng thái."""
        self.text_status.configure(state="normal")
        self.text_status.delete("1.0", "end")
        self.text_status.configure(state="disabled")

    def _set_processing(self, is_processing):
        """Vô hiệu hóa / kích hoạt nút khi đang xử lý."""
        self._processing = is_processing
        state = "disabled" if is_processing else "normal"

        self.btn_choose.configure(state=state)
        self.btn_detect.configure(
            state="disabled" if (is_processing or self.original_image is None)
            else "normal"
        )
        self.btn_manual_detect.configure(
            state="disabled" if (is_processing or self.original_image is None)
            else "normal"
        )
        self.btn_manual_select.configure(
            state="disabled" if (is_processing or self.original_image is None)
            else "normal"
        )
        self.btn_add_noise.configure(
            state="disabled" if (is_processing or self.original_image is None)
            else "normal"
        )
        self.btn_denoise.configure(
            state="disabled" if (is_processing or self.noisy_image is None)
            else "normal"
        )
        self.btn_restore.configure(
            state="disabled" if (is_processing or self.original_image is None)
            else "normal"
        )
        self.btn_compare.configure(
            state="disabled" if (is_processing or self.result is None)
            else "normal"
        )
        self.btn_save.configure(
            state="disabled" if (is_processing or self.result is None)
            else "normal"
        )
        self.btn_csv.configure(
            state="disabled" if (is_processing or not self.csv_rows)
            else "normal"
        )

    def _on_input_mode_changed(self):
        """Discard stale results when the input interpretation changes."""
        self.result = None
        self.noisy_result = None
        self.denoised_result = None
        self.manual_bbox = None
        self._close_cv_windows()
        if self.original_image is not None:
            mode_name = (
                "Ảnh biển đã cắt" if self.image_mode_var.get() == "full_image"
                else "Ảnh toàn xe"
            )
            self._log(f"Đã chuyển sang '{mode_name}'; đã xóa kết quả cũ.")
        self._set_processing(False)
    def _choose_image(self):
        """Mở hộp thoại chọn ảnh."""
        filetypes = [
            ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("Tất cả", "*.*"),
        ]
        initial_dir = config.SAMPLES_DIR if os.path.exists(config.SAMPLES_DIR) else config.PROJECT_DIR
        path = filedialog.askopenfilename(
            title="Chọn ảnh xe",
            initialdir=initial_dir,
            filetypes=filetypes,
        )
        if not path:
            return

        ok = self._load_image_from_path(path)
        if not ok:
            messagebox.showerror("Lỗi", "Không thể nạp ảnh đã chọn.")

    def _load_image_from_path(self, path, auto_detect=False):
        """Nạp ảnh từ đường dẫn và cập nhật trạng thái."""
        if not path or not os.path.exists(path):
            return False

        self._clear_log()
        self._log(f"Đang tải ảnh: {path}")

        img, scale, error = load_image(path)
        if error:
            self._log(f"❌ Lỗi: {error}")
            return False

        # Reset tất cả trạng thái
        self.image_path = path
        self.original_image = img
        self.current_image = img.copy()
        self.noisy_image = None
        self.denoised_image = None
        self.scale_factor = scale
        self.result = None
        self.noisy_result = None
        self.denoised_result = None
        self.manual_bbox = None

        # Đóng cửa sổ OpenCV cũ
        self._close_cv_windows()

        h, w = img.shape[:2]
        self.lbl_path.configure(text=os.path.basename(path))
        self._log(f"✅ Đã tải ảnh: {w}x{h} px")
        if scale < 1.0:
            self._log(f"   (đã resize, hệ số: {scale:.2f})")

        self._set_processing(False)

        if auto_detect:
            self._run_detection()
        return True

    def _toggle_hot_folder(self):
        """Bật/tắt chế độ theo dõi Thư mục nóng (Hot Folder)."""
        if self._hot_folder_running:
            self._stop_hot_folder()
        else:
            self._start_hot_folder()

    def _start_hot_folder(self):
        """Khởi động luồng theo dõi thư mục test_live/."""
        hot_dir = config.HOT_FOLDER_DIR
        os.makedirs(hot_dir, exist_ok=True)
        self._hot_folder_running = True
        self.btn_hot_folder.configure(text="🔴 Đang quét test_live/ (Dừng)")
        self._log(f"\n📂 [Hot Folder] ĐÃ BẬT theo dõi: {hot_dir}")
        self._log("👉 Hãy copy hoặc kéo-thả ảnh xe vào thư mục 'test_live/' để tự động nhận diện!")

        # Ghi nhận các file hiện có
        existing = set()
        if os.path.exists(hot_dir):
            for f in os.listdir(hot_dir):
                if f.lower().endswith(config.SUPPORTED_FORMATS):
                    existing.add(f)
        self._hot_folder_seen = existing

        def watcher():
            import time
            while self._hot_folder_running:
                try:
                    if os.path.exists(hot_dir):
                        current_files = os.listdir(hot_dir)
                        for fname in current_files:
                            ext = os.path.splitext(fname)[1].lower()
                            if ext in config.SUPPORTED_FORMATS and fname not in self._hot_folder_seen:
                                full_p = os.path.join(hot_dir, fname)
                                # Đợi file copy xong
                                time.sleep(0.3)
                                if os.path.exists(full_p) and os.path.getsize(full_p) > 0:
                                    self._hot_folder_seen.add(fname)
                                    self.root.after(0, lambda p=full_p: self._on_hot_folder_file(p))
                except Exception:
                    pass
                time.sleep(0.6)

        self._hot_folder_thread = threading.Thread(target=watcher, daemon=True)
        self._hot_folder_thread.start()

    def _stop_hot_folder(self):
        """Dừng luồng theo dõi thư mục nóng."""
        if self._hot_folder_running:
            self._hot_folder_running = False
            self.btn_hot_folder.configure(text="📂 Thư mục nóng (test_live/)")
            self._log("⏹️ [Hot Folder] Đã dừng theo dõi thư mục nóng.")

    def _on_hot_folder_file(self, file_path):
        """Xử lý khi có ảnh mới xuất hiện trong thư mục test_live/."""
        fname = os.path.basename(file_path)
        self._log(f"\n🔥 [Hot Folder] Phát hiện ảnh mới: {fname}")
        self._load_image_from_path(file_path, auto_detect=True)

    def _open_hot_folder(self):
        """Mở thư mục test_live trên File Explorer của Windows."""
        hot_dir = config.HOT_FOLDER_DIR
        os.makedirs(hot_dir, exist_ok=True)
        try:
            os.startfile(hot_dir)
            self._log(f"📂 Đã mở thư mục: {hot_dir}")
        except Exception as e:
            self._log(f"❌ Không thể mở thư mục: {e}")



    def _restore_original(self):
        """Khôi phục ảnh gốc, xóa nhiễu và kết quả nhiễu/giảm nhiễu."""
        if self.original_image is None:
            return

        self.current_image = self.original_image.copy()
        self.noisy_image = None
        self.denoised_image = None
        self.noisy_result = None
        self.denoised_result = None

        self._close_cv_windows()
        self._log("🔄 Đã khôi phục ảnh gốc.")
        self._set_processing(False)

    def _run_detection(self):
        """Chạy nhận diện tự động (trong worker thread)."""
        if self.original_image is None:
            return
        if self._processing:
            # Nếu đang bận chạy OCR trước đó, lên lịch chạy lại sau 250ms
            self.root.after(250, self._run_detection)
            return

        self._set_processing(True)
        self._log("\n⏳ Đang nhận diện biển số (tự động)...")


        # Xác định trạng thái ảnh đang xử lý
        # Phải xác định TRƯỚC khi vào thread để tránh race condition
        if self.denoised_image is not None:
            target_state = "denoised"
        elif self.noisy_image is not None:
            target_state = "noisy"
        else:
            target_state = "original"

        use_full = self.image_mode_var.get() == "full_image"

        def worker():
            try:
                # Khởi tạo OCR lần đầu
                if not self._ocr_initialized:
                    self.root.after(0, lambda: self._log("  Đang nạp mô hình OCR (lần đầu)..."))
                    self.ocr_engine.initialize()
                    self._ocr_initialized = True
                    self.root.after(0, lambda: self._log(
                        f"  Nạp OCR xong ({self.ocr_engine.load_time:.1f}s)"))

                if use_full:
                    self.root.after(0, lambda: self._log(
                        "  Chế độ: Ảnh biển đã cắt (dùng toàn ảnh làm ROI)"))

                result = process_image(
                    self.current_image,
                    ocr_engine=self.ocr_engine,
                    use_full_image=use_full,
                    save_debug=True,
                )

                # Trả kết quả về luồng GUI, kèm target_state
                self.root.after(0, lambda: self._on_detection_done(
                    result, "full_image" if use_full else "auto", target_state))

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.root.after(0, lambda: self._on_detection_error(f"{e}\n{tb}"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_manual_detection(self):
        """Chạy nhận diện trên vùng thủ công."""
        if self.original_image is None or self._processing:
            return

        # Đọc tọa độ
        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())
            w = int(self.entry_w.get())
            h = int(self.entry_h.get())
        except ValueError:
            self._log("❌ Tọa độ không hợp lệ. Vui lòng nhập số nguyên.")
            return

        if w <= 0 or h <= 0:
            self._log("❌ Chiều rộng và chiều cao phải > 0.")
            return

        self._set_processing(True)
        self.manual_bbox = (x, y, w, h)
        self._log(f"\n⏳ Đang nhận diện vùng thủ công: ({x}, {y}, {w}, {h})...")

        target_state = "original"  # Manual luôn dựa trên ảnh gốc

        def worker():
            try:
                if not self._ocr_initialized:
                    self.root.after(0, lambda: self._log("  Đang nạp mô hình OCR..."))
                    self.ocr_engine.initialize()
                    self._ocr_initialized = True

                result = process_image(
                    self.current_image,
                    ocr_engine=self.ocr_engine,
                    manual_bbox=self.manual_bbox,
                    save_debug=True,
                )

                self.root.after(0, lambda: self._on_detection_done(
                    result, "manual", target_state))

            except Exception as e:
                self.root.after(0, lambda: self._on_detection_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _select_roi_mouse(self):
        """
        Cho người dùng kéo chọn vùng biển trên ảnh bằng chuột.
        Chạy trong thread riêng để không block tkinter mainloop.
        """
        if self.current_image is None:
            return

        self._log("🖱️ Kéo chuột để chọn vùng biển. Nhấn Enter/Space xác nhận, C hủy.")

        # Chuẩn bị ảnh display
        display = self.current_image.copy()
        h, w = display.shape[:2]
        scale = 1.0
        if w > config.MAX_DISPLAY_WIDTH or h > config.MAX_DISPLAY_HEIGHT:
            scale = min(config.MAX_DISPLAY_WIDTH / w, config.MAX_DISPLAY_HEIGHT / h)
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        def do_roi():
            """Chạy selectROI trong thread riêng."""
            try:
                window_name = "Keo chon vung bien - Enter/Space xac nhan, C huy"
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(window_name, 530, 20)
                except Exception:
                    pass
                roi = cv2.selectROI(window_name, display,
                                    showCrosshair=True, fromCenter=False)
                cv2.destroyWindow(window_name)

                if roi[2] > 0 and roi[3] > 0:
                    rx, ry, rw, rh = roi
                    if scale != 1.0:
                        rx = int(rx / scale)
                        ry = int(ry / scale)
                        rw = int(rw / scale)
                        rh = int(rh / scale)

                    # Cập nhật GUI từ luồng chính
                    def update_entries():
                        self.entry_x.delete(0, "end")
                        self.entry_x.insert(0, str(rx))
                        self.entry_y.delete(0, "end")
                        self.entry_y.insert(0, str(ry))
                        self.entry_w.delete(0, "end")
                        self.entry_w.insert(0, str(rw))
                        self.entry_h.delete(0, "end")
                        self.entry_h.insert(0, str(rh))
                        self._log(f"✅ Đã chọn vùng: ({rx}, {ry}, {rw}, {rh})")

                    self.root.after(0, update_entries)
                else:
                    self.root.after(0, lambda: self._log("⚠️ Đã hủy chọn vùng."))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"❌ Lỗi chọn vùng: {e}"))

        threading.Thread(target=do_roi, daemon=True).start()

    def _on_detection_done(self, result, method, target_state="original"):
        """
        Callback khi nhận diện xong (chạy trên luồng GUI).

        Tham số:
            result: kết quả pipeline
            method: "auto" / "manual" / "region"
            target_state: "original" / "noisy" / "denoised"
                Xác định rõ ràng kết quả thuộc trạng thái nào.
        """
        if result.get("error"):
            self._log(f"⚠️ {result['error']}")
        if result.get("debug_dir"):
            self._log(f"   Debug: {result['debug_dir']}")

        if result["n_plates"] == 0:
            self._log("⚠️ Không tìm thấy biển số nào (do ảnh quá tối hoặc quá nhiễu).")
            # Vẫn cập nhật hiển thị ảnh bị nhiễu/tối lên màn hình kèm thông báo
            display_img = self.current_image.copy()
            cv2.putText(
                display_img,
                "Khong tim thay bien so (Anh qua toi / nhieu)",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            wname = show_result_nonblocking(display_img, config.WINDOW_TITLE)
            if wname not in self._cv_windows:
                self._cv_windows.append(wname)
            self._set_processing(False)
            return


        # Lưu kết quả theo trạng thái được chỉ định rõ ràng
        if target_state == "denoised":
            self.denoised_result = result
        elif target_state == "noisy":
            self.noisy_result = result
        else:
            # "original" hoặc mặc định
            self.result = result

        # Hiển thị kết quả
        self._log(f"\n📋 Tìm thấy {result['n_plates']} biển số:")
        self._log(f"   Phương pháp: {'Thủ công' if method == 'manual' else 'Tự động'}")
        self._log(f"   Tổng thời gian: {result['total_time']:.3f}s")

        plates_info = []
        for plate in result["plates"]:
            self._log(f"\n   Biển {plate.get('plate_idx', 0) + 1}:")
            self._log(f"   ├── Biển số : {plate.get('plate_text', 'N/A')}")
            self._log(f"   ├── Màu nền : {plate.get('color', 'N/A')}")
            self._log(f"   ├── Bố cục  : {plate.get('layout', 'N/A')}")
            self._log(f"   ├── Điểm OCR: {plate.get('ocr_confidence', 0):.3f}")
            self._log(f"   └── Biến thể: {plate.get('best_variant', 'N/A')}")

            plates_info.append({
                "bbox": plate["bbox"],
                "plate_text": plate.get("plate_text", "Không đọc được"),
                "color": plate.get("color", "Không xác định"),
                "layout": plate.get("layout", "Không xác định"),
                "ocr_confidence": plate.get("ocr_confidence", 0),
            })

            # In tóm tắt ra terminal
            print_summary(plate, method)

            # Thêm vào CSV
            image_name = os.path.basename(self.image_path) if self.image_path else "unknown"

            # Xác định trạng thái ảnh cho CSV dựa trên target_state
            state_map = {
                "original": "gốc",
                "noisy": "nhiễu",
                "denoised": "giảm nhiễu",
            }
            state = state_map.get(target_state, "gốc")

            csv_row = format_result_for_csv(image_name, plate, state)
            self.csv_rows.append(csv_row)

        # Vẽ và hiển thị kết quả
        img_to_draw = self.current_image
        annotated = draw_results(img_to_draw, plates_info)
        wname = show_result_nonblocking(annotated, config.WINDOW_TITLE)
        if wname not in self._cv_windows:
            self._cv_windows.append(wname)

        self._set_processing(False)

    def _on_detection_error(self, error_msg):
        """Callback khi nhận diện lỗi."""
        self._log(f"❌ Lỗi: {error_msg}")
        self._set_processing(False)

    # =================================================================
    # PHẦN 3: TẠO NHIỄU VÀ GIẢM NHIỄU
    # =================================================================

    def _on_noise_type_changed(self, event=None):
        """Cập nhật giao diện khi đổi loại nhiễu."""
        self._update_noise_hint()

    def _update_noise_hint(self, *args):
        """Cập nhật gợi ý thông số nhiễu."""
        noise_type = self.noise_type_var.get()
        val = self.noise_scale.get()

        if noise_type == "gaussian":
            self.lbl_noise_hint.configure(text=f"(sigma={val:.1f})")
        elif noise_type == "salt_pepper":
            ratio = val / 100.0
            self.lbl_noise_hint.configure(text=f"(tỷ lệ={ratio:.2f})")
        elif noise_type == "blur":
            ksize = int(val)
            if ksize % 2 == 0:
                ksize += 1
            ksize = max(1, ksize)
            self.lbl_noise_hint.configure(text=f"(kernel={ksize})")
        elif noise_type in ("brightness", "night"):
            factor = max(0.05, val / 100.0)
            self.lbl_noise_hint.configure(text=f"(độ sáng={factor:.2f})")

    def _add_noise(self):
        """Tạo nhiễu cho ảnh."""
        if self.original_image is None or self._processing:
            return

        noise_type = self.noise_type_var.get()
        val = self.noise_scale.get()
        scope = self.noise_scope_var.get()

        # Lấy seed
        seed_text = self.entry_seed.get().strip()
        seed = int(seed_text) if seed_text.isdigit() else None

        # Xây dựng tham số
        params = {"seed": seed}
        if noise_type == "gaussian":
            params["sigma"] = val
        elif noise_type == "salt_pepper":
            params["ratio"] = val / 100.0
        elif noise_type == "blur":
            ksize = int(val)
            if ksize % 2 == 0:
                ksize += 1
            params["ksize"] = max(1, ksize)
        elif noise_type in ("brightness", "night"):
            params["factor"] = max(0.05, val / 100.0)

        # Xác định ROI nếu scope == "roi"
        roi_bbox = None
        if scope == "roi":
            if self.result and self.result["plates"]:
                roi_bbox = self.result["plates"][0]["bbox"]
            elif self.manual_bbox:
                roi_bbox = self.manual_bbox
            else:
                self._log("⚠️ Chưa có vùng biển. Hãy nhận diện trước hoặc chọn 'full'.")
                return

        # Tạo nhiễu từ ảnh GỐC (không cộng dồn)
        self.noisy_image = apply_noise(
            self.original_image, noise_type, params, scope, roi_bbox
        )
        self.current_image = self.noisy_image.copy()
        self.denoised_image = None
        self.noisy_result = None
        self.denoised_result = None

        # Phân biệt blur và nhiễu trong log
        type_name = {
            "gaussian": "Gaussian noise",
            "salt_pepper": "Muối tiêu",
            "blur": "Gaussian blur (làm mờ)",
            "brightness": "Giảm sáng mô phỏng ban đêm",
        }

        self._log(f"\n🔊 Đã tạo: {type_name.get(noise_type, noise_type)}")
        self._log(f"   Phạm vi: {'Toàn ảnh' if scope == 'full' else 'Vùng biển'}")
        self._log(f"   Thông số: {params}")
        if seed is not None:
            self._log(f"   Seed: {seed}")

        # Hiển thị ảnh nhiễu
        wname = show_result_nonblocking(self.noisy_image, "Ảnh nhiễu")
        if wname not in self._cv_windows:
            self._cv_windows.append(wname)

        self._set_processing(False)

    def _on_key_press(self, event):
        """Bắt sự kiện gõ phím từ cửa sổ Tkinter."""
        if isinstance(event.widget, (tk.Entry, ttk.Entry)):
            return
        if not event.char:
            return
        self._handle_shortcut(event.char.lower())

    def _handle_shortcut(self, ch):
        """Xử lý phím tắt Live Demo (từ cả Tkinter và OpenCV)."""
        if ch == 'b':
            self._live_apply_noise("blur", {"ksize": 15}, "Làm mờ ảnh (Gaussian Blur, k=15)")
        elif ch == 'n':
            self._live_apply_noise("gaussian", {"sigma": 45, "seed": 42}, "Nhiễu hạt cảm biến (Gaussian, sigma=45)")
        elif ch == 's':
            self._live_apply_noise("salt_pepper", {"ratio": 0.10, "seed": 42}, "Nhiễu muối tiêu (Salt & Pepper, 10%)")
        elif ch == 'd':
            self._live_apply_noise("brightness", {"factor": 0.45}, "Giảm sáng ban đêm (Night Mode, 45% sáng)")
        elif ch == 'g':
            if self.noisy_image is not None or self.current_image is not None:
                self._log("\n⚡ [Live Denoise] Phím tắt 'G': Áp dụng giảm nhiễu và nhận diện lại...")
                self._apply_denoise()
        elif ch == 'r':
            if self.original_image is not None:
                self._log("\n⚡ [Live Reset] Phím tắt 'R': Khôi phục ảnh gốc và nhận diện lại...")
                self._restore_original()
                self._run_detection()

    def _live_apply_noise(self, noise_type, params, label):
        """Tạo nhiễu trực tiếp và lập tức chạy qua pipeline nhận diện."""
        if self.original_image is None:
            self._log("⚠️ Vui lòng chọn ảnh trước khi dùng phím tắt tạo nhiễu.")
            return

        self._log(f"\n⚡ [Live Noise Injection] Kích hoạt: {label}")
        self.noise_type_var.set(noise_type if noise_type != "night" else "brightness")
        self.noise_scope_var.set("full")
        self.noisy_image = apply_noise(self.original_image, noise_type, params=params, scope="full")
        self.current_image = self.noisy_image.copy()
        self.denoised_image = None
        self.noisy_result = None
        self.denoised_result = None

        self.btn_restore.configure(state="normal")
        self.btn_denoise.configure(state="normal")
        self.btn_save.configure(state="normal")

        # Cập nhật hiển thị ảnh tức thì lên màn hình (không độ trễ)
        show_result_nonblocking(self.current_image, config.WINDOW_TITLE)
        if config.WINDOW_TITLE not in self._cv_windows:
            self._cv_windows.append(config.WINDOW_TITLE)

        # Tự động nhận diện lại ngay lập tức
        self._run_detection()



    def _run_detection_on_current(self, state, noise_type=None, noise_params=None,
                                   scope=None):
        """
        Chạy nhận diện trên current_image (nhiễu hoặc giảm nhiễu).

        Tham số:
            state: "noisy" / "denoised" - xác định kết quả sẽ lưu vào đâu
        """
        self._set_processing(True)

        def worker():
            try:
                if not self._ocr_initialized:
                    self.ocr_engine.initialize()
                    self._ocr_initialized = True

                if scope == "roi" and state == "noisy":
                    # Chế độ vùng biển: chỉ chạy OCR trên vùng biển nhiễu
                    roi_bbox = None
                    if self.result and self.result["plates"]:
                        roi_bbox = self.result["plates"][0]["bbox"]
                    elif self.manual_bbox:
                        roi_bbox = self.manual_bbox

                    if roi_bbox:
                        x, y, w, h = roi_bbox
                        plate_region = self.current_image[y:y+h, x:x+w].copy()
                        result_single = process_plate_region(
                            plate_region, roi_bbox, self.ocr_engine
                        )
                        result = {
                            "plates": [result_single],
                            "total_time": result_single["processing_time"],
                            "n_plates": 1,
                            "detection_method": "region",
                        }
                    else:
                        result = process_image(self.current_image,
                                               ocr_engine=self.ocr_engine)
                else:
                    result = process_image(self.current_image,
                                           ocr_engine=self.ocr_engine)

                # Truyền target_state xuống callback để lưu đúng chỗ
                self.root.after(0, lambda: self._on_detection_done(
                    result, result.get("detection_method", "auto"), state))

            except Exception as e:
                self.root.after(0, lambda: self._on_detection_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _run_denoise(self):
        """Giảm nhiễu và nhận diện lại."""
        if self.noisy_image is None or self._processing:
            return

        method = self.denoise_type_var.get()
        ksize = int(self.denoise_scale.get())
        if ksize % 2 == 0:
            ksize += 1

        params = {}
        if method in ("median", "gaussian"):
            params["ksize"] = ksize
        elif method == "bilateral":
            params["d"] = ksize
        elif method == "nlm":
            params["h"] = ksize

        self._log(f"\n🔇 Giảm nhiễu: {method}, thông số: {params}")

        self.denoised_image = apply_denoise(self.noisy_image, method, params)
        self.current_image = self.denoised_image.copy()

        # Hiển thị ảnh giảm nhiễu
        wname = show_result_nonblocking(self.denoised_image, "Ảnh giảm nhiễu")
        if wname not in self._cv_windows:
            self._cv_windows.append(wname)

        # Nhận diện lại
        self._log("⏳ Đang nhận diện lại trên ảnh giảm nhiễu...")
        self._run_detection_on_current("denoised")

    # =================================================================
    # PHẦN 4: SO SÁNH VÀ LƯU
    # =================================================================

    def _show_comparison(self):
        """Hiển thị ảnh so sánh gốc / nhiễu / giảm nhiễu."""
        images = OrderedDict()

        if self.original_image is not None:
            # Vẽ kết quả lên ảnh gốc
            if self.result and self.result["plates"]:
                info = [{
                    "bbox": p["bbox"],
                    "plate_text": p.get("plate_text", ""),
                    "color": p.get("color", ""),
                    "layout": p.get("layout", ""),
                    "ocr_confidence": p.get("ocr_confidence", 0),
                } for p in self.result["plates"]]
                images["Ảnh gốc"] = draw_results(self.original_image, info)
            else:
                images["Ảnh gốc"] = self.original_image

        if self.noisy_image is not None:
            if self.noisy_result and self.noisy_result["plates"]:
                info = [{
                    "bbox": p["bbox"],
                    "plate_text": p.get("plate_text", ""),
                    "color": p.get("color", ""),
                    "layout": p.get("layout", ""),
                    "ocr_confidence": p.get("ocr_confidence", 0),
                } for p in self.noisy_result["plates"]]
                images["Ảnh nhiễu"] = draw_results(self.noisy_image, info)
            else:
                images["Ảnh nhiễu"] = self.noisy_image

        if self.denoised_image is not None:
            if self.denoised_result and self.denoised_result["plates"]:
                info = [{
                    "bbox": p["bbox"],
                    "plate_text": p.get("plate_text", ""),
                    "color": p.get("color", ""),
                    "layout": p.get("layout", ""),
                    "ocr_confidence": p.get("ocr_confidence", 0),
                } for p in self.denoised_result["plates"]]
                images["Ảnh giảm nhiễu"] = draw_results(self.denoised_image, info)
            else:
                images["Ảnh giảm nhiễu"] = self.denoised_image

        if len(images) < 2:
            self._log("⚠️ Cần ít nhất 2 trạng thái ảnh để so sánh.")
            return

        comparison = create_comparison_image(images)
        if comparison is not None:
            wname = show_result_nonblocking(comparison, "So sánh kết quả")
            if wname not in self._cv_windows:
                self._cv_windows.append(wname)

        # Log thông tin so sánh
        self._log("\n📊 SO SÁNH KẾT QUẢ:")
        for state_name, res in [("Gốc", self.result),
                                 ("Nhiễu", self.noisy_result),
                                 ("Giảm nhiễu", self.denoised_result)]:
            if res and res["plates"]:
                p = res["plates"][0]
                self._log(f"  {state_name}: {p.get('plate_text', 'N/A')} | "
                          f"{p.get('color', 'N/A')} | {p.get('layout', 'N/A')} | "
                          f"OCR={p.get('ocr_confidence', 0):.3f} | "
                          f"t={p.get('processing_time', 0):.3f}s")

    def _save_results(self):
        """Lưu ảnh kết quả."""
        if self.result is None:
            self._log("⚠️ Chưa có kết quả để lưu.")
            return

        save_dir = filedialog.askdirectory(title="Chọn thư mục lưu")
        if not save_dir:
            return

        saved = []
        base = os.path.splitext(os.path.basename(self.image_path or "result"))[0]

        # Lưu ảnh có khung và nhãn
        if self.result and self.result["plates"]:
            info = [{
                "bbox": p["bbox"],
                "plate_text": p.get("plate_text", ""),
                "color": p.get("color", ""),
                "layout": p.get("layout", ""),
            } for p in self.result["plates"]]
            annotated = draw_results(self.original_image, info)
            path = os.path.join(save_dir, f"{base}_result.jpg")
            ok, err = save_image(annotated, path)
            if ok:
                saved.append(path)

            # Lưu vùng biển cắt
            for i, p in enumerate(self.result["plates"]):
                if "plate_bgr" in p:
                    ppath = os.path.join(save_dir, f"{base}_plate_{i}.jpg")
                    ok, _ = save_image(p["plate_bgr"], ppath)
                    if ok:
                        saved.append(ppath)

        # Lưu ảnh nhiễu
        if self.noisy_image is not None:
            path = os.path.join(save_dir, f"{base}_noisy.jpg")
            ok, _ = save_image(self.noisy_image, path)
            if ok:
                saved.append(path)

        # Lưu ảnh giảm nhiễu
        if self.denoised_image is not None:
            path = os.path.join(save_dir, f"{base}_denoised.jpg")
            ok, _ = save_image(self.denoised_image, path)
            if ok:
                saved.append(path)

        self._log(f"\n💾 Đã lưu {len(saved)} file vào {save_dir}")
        for s in saved:
            self._log(f"   {os.path.basename(s)}")

    def _export_csv(self):
        """Xuất dữ liệu CSV."""
        if not self.csv_rows:
            self._log("⚠️ Chưa có dữ liệu để xuất.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Lưu file CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialdir=config.OUTPUT_DIR,
        )
        if not filepath:
            return

        ok, msg = export_csv(self.csv_rows, filepath)
        if ok:
            self._log(f"✅ Đã xuất CSV: {msg}")
        else:
            self._log(f"❌ {msg}")

    # =================================================================
    # PHẦN 5: QUẢN LÝ CỬA SỔ VÀ TÀI NGUYÊN
    # =================================================================

    def _poll_cv(self):
        """
        Gọi cv2.waitKey định kỳ để cửa sổ OpenCV phản hồi.
        KHÔNG dùng vòng chờ OpenCV chặn tkinter.
        """
        if self._cv_windows:
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    self._close_cv_windows()
                elif key in (ord('b'), ord('B')):
                    self.root.after(0, lambda: self._handle_shortcut('b'))
                elif key in (ord('n'), ord('N')):
                    self.root.after(0, lambda: self._handle_shortcut('n'))
                elif key in (ord('s'), ord('S')):
                    self.root.after(0, lambda: self._handle_shortcut('s'))
                elif key in (ord('d'), ord('D')):
                    self.root.after(0, lambda: self._handle_shortcut('d'))
                elif key in (ord('g'), ord('G')):
                    self.root.after(0, lambda: self._handle_shortcut('g'))
                elif key in (ord('r'), ord('R')):
                    self.root.after(0, lambda: self._handle_shortcut('r'))
            except Exception:
                pass

        # Gọi lại sau 50ms
        self.root.after(50, self._poll_cv)

    def _close_cv_windows(self):
        """Đóng tất cả cửa sổ OpenCV."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._cv_windows.clear()

    def _on_close(self):
        """Xử lý khi đóng cửa sổ chính."""
        self._stop_hot_folder()
        self._close_cv_windows()
        self.root.destroy()

