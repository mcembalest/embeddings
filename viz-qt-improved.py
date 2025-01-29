import sys
import os
import numpy as np
import threading
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar,
    QDialog, QFormLayout, QLineEdit, QPushButton, QHBoxLayout, QMessageBox,
    QMenuBar, QToolBar, QFileDialog, QStatusBar, QSplitter, QSizePolicy,
    QTextEdit, QPlainTextEdit, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot, QThread, QEvent
from PySide6.QtGui import QAction, QIcon, QPainter, QPixmap, QLinearGradient, QColor, QPalette

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import umap

def set_dark_theme(app: QApplication) -> None:
    """
    Applies a custom dark palette to the given QApplication instance.
    """
    app.setStyle("Fusion")
    dark_palette = QPalette()
    
    # Base colors
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    
    # Highlight
    dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)

# -------------------------------------------------------------------------
# Animation widget
# -------------------------------------------------------------------------
class AnimatedPattern(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.pattern_counter = 0
        
        # Load and scale the image
        self.image = QPixmap("image.png")
        
        # Pre-calculate pattern coordinates for each frame
        self.patterns = []
        for frame in range(4):
            coords = []
            for i in range(32):
                for j in range(32):
                    if (i + j + frame) % 4 == 0:
                        coords.append((i, j))
            self.patterns.append(coords)
        
        # Create timer for animation (4x per second = 250ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pattern)
        self.timer.start(75)
    
    def update_pattern(self):
        self.pattern_counter = (self.pattern_counter + 1) % 4
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        
        # Draw the scaled background image
        scaled_image = self.image.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (width - scaled_image.width()) // 2
        y = (height - scaled_image.height()) // 2
        painter.drawPixmap(x, y, scaled_image)
        
        # Calculate cell sizes based on actual dimensions
        cell_width = width / 32
        cell_height = height / 32
        
        # Set up semi-transparent overlay
        painter.setOpacity(0.7)
        
        # Draw pre-calculated pattern for current frame with a gradient effect
        for i, j in self.patterns[self.pattern_counter]:
            gradient = QLinearGradient(
                i * cell_width, j * cell_height,
                (i + 1) * cell_width, (j + 1) * cell_height
            )
            gradient.setColorAt(0, QColor(0, 0, 0, 200))
            gradient.setColorAt(1, QColor(0, 0, 0, 100))
            painter.fillRect(
                i * cell_width,
                j * cell_height,
                cell_width,
                cell_height,
                gradient
            )

# -------------------------------------------------------------------------
# Basic chat output widget
# -------------------------------------------------------------------------
class ChatOutput(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText(">>>")
        
        # Timer for simulating streaming output
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.stream_next_char)
        self.current_message = ""
        self.current_pos = 0
        self.displayed_text = ""
        
    def process_input(self, text):
        # Generate some random response text
        responses = [
            "Interesting observation! Let me analyze that further...",
            "I see what you mean. Here's what I think...", 
            "That's a fascinating point. Consider this perspective...",
            "Based on the data, I would suggest...",
            "Let me break this down step by step..."
        ]
        self.current_message = random.choice(responses)
        self.current_pos = 0
        self.displayed_text = ""
        
        # Start streaming the response
        self.stream_timer.start(10)
        
    def stream_next_char(self):
        if self.current_pos < len(self.current_message):
            self.displayed_text += self.current_message[self.current_pos]
            self.setPlainText(self.displayed_text)
            self.current_pos += 1
        else:
            self.stream_timer.stop()
            self.displayed_text += "\n>>>"
            self.setPlainText(self.displayed_text)

# -------------------------------------------------------------------------
# UMAP Settings Dialog
# -------------------------------------------------------------------------
class UMAPSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UMAP Settings")
        self.setModal(True)
        self.layout = QFormLayout(self)
        
        self.n_neighbors_input = QLineEdit(self)
        self.n_neighbors_input.setText("15")
        self.layout.addRow("Number of Neighbors:", self.n_neighbors_input)
        
        self.min_dist_input = QLineEdit(self)
        self.min_dist_input.setText("0.1")
        self.layout.addRow("Minimum Distance:", self.min_dist_input)
        
        self.n_components_input = QLineEdit(self)
        self.n_components_input.setText("2")
        self.layout.addRow("Number of Components:", self.n_components_input)
        
        self.metric_input = QLineEdit(self)
        self.metric_input.setText("euclidean")
        self.layout.addRow("Metric:", self.metric_input)
        
        # Buttons
        self.buttons_layout = QHBoxLayout()
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.cancel_button)
        self.layout.addRow(self.buttons_layout)
    
    def save_settings(self):
        try:
            self.n_neighbors = int(self.n_neighbors_input.text())
            self.min_dist = float(self.min_dist_input.text())
            self.n_components = int(self.n_components_input.text())
            self.metric = self.metric_input.text()
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid parameter values.")

# -------------------------------------------------------------------------
# Simple Matplotlib canvas for showing UMAP scatter
# -------------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        
        # Add tight_layout to prevent label cutoff
        fig.tight_layout()
        
        self.scatter = None  # Initialize scatter
        
        # Remove axes
        self.axes.set_xticks([])
        self.axes.set_yticks([])
    
    def plot_umap(self, points_2d, epoch=None):
        """Update scatter plot data with optional epoch info."""
        self.axes.clear()  # Clear previous plot
        self.scatter = self.axes.scatter(points_2d[:, 0], points_2d[:, 1], s=10, alpha=0.7)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.figure.tight_layout()
        self.draw()

# -------------------------------------------------------------------------
# Custom UMAP Callback for partial progress updates
# -------------------------------------------------------------------------
class UMAPCallback(QObject):
    progress_updated = Signal(int)
    embedding_updated = Signal(object, int)
    
    def __init__(self, total_epochs, reducer, update_interval=5):
        super().__init__()
        self.total_epochs = total_epochs
        self.current = 0
        self.reducer = reducer
        self.update_interval = update_interval
    
    def __call__(self, *args, **kwargs):
        """Called by UMAP for each epoch"""
        self.current += 1
        
        # Emit progress signal from the callback thread
        progress = int((self.current / self.total_epochs) * 100)
        QApplication.instance().postEvent(self, QEvent(QEvent.Type.User))
        self.progress_updated.emit(progress)
        
        # Emit embedding update at intervals
        if self.current % self.update_interval == 0 or self.current == self.total_epochs:
            embedding = self.reducer.embedding_
            if embedding is not None:
                self.embedding_updated.emit(np.copy(embedding), self.current)

# -------------------------------------------------------------------------
# Drop Label: handles file dropping, progress bar, and triggers UMAP
# -------------------------------------------------------------------------
class DropLabel(QLabel):
    def __init__(self, canvas, parent=None):
        super().__init__("Drop a file here\nor click to load", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 8px;
                background-color: #282828;
                font-size: 16px;
                padding: 20px;
                min-height: 10px;
            }
            QLabel:hover {
                background-color: #686868;
                border-color: #888;
                cursor: pointer;
            }
        """)
        self.setAcceptDrops(True)
        self.canvas = canvas

        # Single progress bar used for both indeterminate & determinate
        self.progress = QProgressBar(self)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.hide()
        
        # self.status_label = QLabel(self)
        # self.status_label.setAlignment(Qt.AlignCenter)
        # self.status_label.hide()
        
        layout = QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(self.progress)
        layout.addStretch()
        self._n_epochs = 0
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)

        self.current = 0

    def mousePressEvent(self, event):
        """Clicking opens file dialog in the MainWindow."""
        if event.button() == Qt.LeftButton:
            # Find the MainWindow by traversing up
            parent = self
            while parent is not None and not isinstance(parent, QMainWindow):
                parent = parent.parent()
            if parent is not None:
                parent.open_file()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith(".npy") for url in urls):
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(".npy"):
                    self.process_file(file_path)

    def load_file(self, file_path):
        """Called by main window open dialog."""
        if file_path:
            self.process_file(file_path)

    def process_file(self, file_path):
        """Central method: loads file, checks shape, and runs UMAP."""
        if not file_path.lower().endswith(".npy"):
            self.show_error("Not an .npy file.")
            return

        self.setText("")
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.show()

        # Load data
        try:
            data = np.load(file_path)
        except Exception as e:
            self.show_error(f"Failed to load file: {str(e)}")
            return

        if data.ndim != 2 or data.shape[0] < 2:
            self.show_error("Data shape not suitable for UMAP.")
            return

        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # Retrieve settings (same as before)
        main_window = self._find_main_window()
        umap_settings = getattr(main_window, 'umap_settings', {}) if main_window else {}

        self._n_epochs = 200  # Track total epochs
        umap_reducer = umap.UMAP(
            n_neighbors=umap_settings.get("n_neighbors", 15),
            min_dist=umap_settings.get("min_dist", 0.1),
            n_components=umap_settings.get("n_components", 2),
            metric=umap_settings.get("metric", "euclidean"),
            random_state=42,
            verbose=True,
            n_epochs=self._n_epochs
        )
        
        # Create callback and connect signals
        callback = UMAPCallback(self._n_epochs, umap_reducer, update_interval=5)
        
        # Use moveToThread to ensure signals work across threads
        thread = QThread()
        callback.moveToThread(thread)
        
        # Connect signals using Qt.QueuedConnection to handle cross-thread signals
        callback.progress_updated.connect(self.update_progress, Qt.QueuedConnection)
        callback.embedding_updated.connect(
            lambda emb, epoch: self.on_embedding_updated(emb, epoch), 
            Qt.QueuedConnection
        )
        
        # Start UMAP in background thread
        threading.Thread(
            target=self.run_umap, 
            args=(umap_reducer, data, callback), 
            daemon=True
        ).start()
        
    def run_umap(self, reducer, data, callback):
        """UMAP in background thread."""
        try:
            reducer.fit(data, callbacks=[callback])
            # Force final embedding update
            self.on_embedding_updated(reducer.embedding_, reducer.n_epochs)
        except Exception as e:
            print(f"Error: {str(e)}")

    @Slot(int)
    def update_progress(self, value):
        """Update progress bar in GUI thread"""
        self.progress.setValue(value)

    @Slot(object, int)
    def on_embedding_updated(self, embedding, epoch):
        """Handle embedding updates in GUI thread"""
        self.canvas.plot_umap(embedding, epoch=epoch)
        print("epoch", epoch)
        if epoch == self._n_epochs: 
            self.reset_label()

    def reset_label(self):
        """Hide progress UI and restore default drop-text."""
        self.progress.hide()
        self.setText("Drop a file here\nor click to load")

    def show_error(self, message):
        """Handle errors uniformly."""
        self.setText(message)
        self.canvas.axes.clear()
        self.canvas.axes.text(
            0.5, 0.5,
            message,
            horizontalalignment='center',
            verticalalignment='center',
            transform=self.canvas.axes.transAxes
        )
        self.canvas.draw()
        self.progress.hide()

    def _find_main_window(self):
        """Helper to find the MainWindow up the hierarchy."""
        parent = self
        while parent is not None and not isinstance(parent, QMainWindow):
            parent = parent.parent()
        return parent

# -------------------------------------------------------------------------
# Two views
# -------------------------------------------------------------------------
class MinimalView(QWidget):
    def __init__(self, parent=None, canvas=None, drop_label=None):
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        self.canvas = canvas
        self.drop_label = drop_label
        
        # Add widgets directly to layout
        if self.drop_label:
            self.drop_label.setMinimumHeight(250)  # Adjust this value as needed
            self.drop_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            layout.addWidget(self.drop_label)
            self.drop_label.show()
        
        if self.canvas:
            layout.addWidget(self.canvas)
            self.canvas.show()
        
        # Make sure this widget is visible
        self.show()
        
        print(f"\nMinimalView state:")
        print(f"- Parent: {self.parent()}")
        print(f"- Visible: {self.isVisible()}")
        print(f"- Drop label visible: {self.drop_label.isVisible() if self.drop_label else 'No drop label'}")
        print(f"- Canvas visible: {self.canvas.isVisible() if self.canvas else 'No canvas'}")

class MaximalView(QWidget):
    def __init__(self, parent=None, canvas=None, drop_label=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Use shared components passed from MainWindow
        self.canvas = canvas
        self.drop_label = drop_label
        
        main_splitter = QSplitter(Qt.Vertical)
        top_splitter = QSplitter(Qt.Horizontal) 
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Create top left widget and layout
        top_left_widget = QWidget()
        top_left_layout = QVBoxLayout(top_left_widget)
        
        # Add drop label to top left if it exists
        if self.drop_label:
            top_left_layout.addWidget(self.drop_label)
        
        # Text input container
        text_input_container = QWidget()
        text_input_layout = QHBoxLayout(text_input_container)
        text_input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Message Vizier")
        self.text_input.setMaximumHeight(150)
        # Connect key press event
        self.text_input.keyPressEvent = self.handle_key_press
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setMaximumHeight(20)
        
        text_input_layout.addWidget(self.text_input)
        text_input_layout.addWidget(self.send_button)
        
        top_left_layout.addWidget(text_input_container)
        
        # Rest of the layout setup
        # Top right: animated pattern
        self.pattern_widget = AnimatedPattern()
        
        # Bottom left: chat output
        self.chat_output = ChatOutput()
        
        top_splitter.addWidget(top_left_widget)
        top_splitter.addWidget(self.pattern_widget)
        
        bottom_splitter.addWidget(self.chat_output)
        bottom_splitter.addWidget(self.canvas)
        
        
        top_splitter.setSizes([1000, 1000])
        bottom_splitter.setSizes([1000, 1000])
        
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setSizes([1000, 1000])
        
        top_left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pattern_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        top_splitter.splitterMoved.connect(
            lambda pos, index: bottom_splitter.moveSplitter(pos, index))
        bottom_splitter.splitterMoved.connect(
            lambda pos, index: top_splitter.moveSplitter(pos, index))
        
        layout.addWidget(main_splitter)

        # Remove the recursive splitter connections
        # Instead, store initial proportions
        self.initial_sizes = [1000, 1000]
        top_splitter.setSizes(self.initial_sizes)
        bottom_splitter.setSizes(self.initial_sizes)
        
        # Optional: Connect splitters to maintain proportions without recursion
        def update_proportions(pos, index):
            sender = self.sender()
            if sender == top_splitter:
                total_width = sum(bottom_splitter.sizes())
                ratio = pos / sum(top_splitter.sizes())
                bottom_splitter.setSizes([int(total_width * ratio), int(total_width * (1-ratio))])
            else:
                total_width = sum(top_splitter.sizes())
                ratio = pos / sum(bottom_splitter.sizes())
                top_splitter.setSizes([int(total_width * ratio), int(total_width * (1-ratio))])
                
        top_splitter.splitterMoved.connect(update_proportions)
        bottom_splitter.splitterMoved.connect(update_proportions)
    
    def handle_key_press(self, event):
        # Check for Shift+Enter or Ctrl+Enter to send message
        if (event.key() == Qt.Key_Return and 
            (event.modifiers() & Qt.ShiftModifier or event.modifiers() & Qt.ControlModifier)):
            self.send_message()
        # Regular Enter creates newline
        elif event.key() == Qt.Key_Return:
            QTextEdit.keyPressEvent(self.text_input, event)
        else:
            QTextEdit.keyPressEvent(self.text_input, event)

    def send_message(self):
        message = self.text_input.toPlainText().strip()
        if message:
            # Add user message to chat
            self.chat_output.appendPlainText(f"\nYou: {message}\n")
            # Process message and get response
            self.chat_output.process_input(message)
            # Clear input
            self.text_input.clear()

# -------------------------------------------------------------------------
# Main Window
# -------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vizier")
        self.setWindowIcon(QIcon("icon.png"))
        
        # Create stacked widget first
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create minimal view first with its own canvas and drop label
        self.minimal_canvas = MplCanvas(width=8, height=8, dpi=100)
        self.minimal_drop_label = DropLabel(canvas=self.minimal_canvas)
        self.minimal_view = MinimalView(
            parent=self.stacked_widget,
            canvas=self.minimal_canvas,
            drop_label=self.minimal_drop_label
        )
        
        # Create maximal view with its own canvas and drop label
        self.maximal_canvas = MplCanvas(width=8, height=8, dpi=100)
        self.maximal_drop_label = DropLabel(canvas=self.maximal_canvas)
        self.maximal_view = MaximalView(
            parent=self.stacked_widget,
            canvas=self.maximal_canvas,
            drop_label=self.maximal_drop_label
        )
        
        # Add views to stacked widget
        self.stacked_widget.addWidget(self.minimal_view)
        self.stacked_widget.addWidget(self.maximal_view)
        
        # Create rest of UI
        self.create_menu_bar()
        self.create_tool_bar()
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        # Start with minimal view and make sure it's visible
        self.minimal_view.show()
        self.stacked_widget.setCurrentWidget(self.minimal_view)
        
        self.resize(400, 600)
        
        # print(f"- Final canvas parent: {self.canvas.parent()}")
        # print(f"- Final drop_label parent: {self.drop_label.parent()}")
        # Default UMAP settings
        self.umap_settings = {}

    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View Menu
        view_menu = menu_bar.addMenu("View")
        toggle_view_action = QAction("Toggle View", self)
        toggle_view_action.triggered.connect(self.toggle_view)
        toggle_view_action.setShortcut("Ctrl+T")
        view_menu.addAction(toggle_view_action)
        
        # Settings Menu
        settings_menu = menu_bar.addMenu("Settings")
        umap_settings_action = QAction("UMAP Parameters", self)
        umap_settings_action.triggered.connect(self.open_umap_settings)
        settings_menu.addAction(umap_settings_action)

    def create_tool_bar(self):
        tool_bar = QToolBar("Main Toolbar")
        self.addToolBar(tool_bar)

        # Logo
        logo_label = QLabel()
        logo_pixmap = QIcon("icon.png").pixmap(36, 36)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setContentsMargins(10, 10, 10, 10)
        tool_bar.addWidget(logo_label)
        
        tool_bar.addSeparator()
        
        # Open
        open_icon = QIcon.fromTheme("document-open")
        open_action = QAction(open_icon, "Open", self)
        open_action.setStatusTip("Open File")
        open_action.setToolTip("")  
        open_action.triggered.connect(self.open_file)
        tool_bar.addAction(open_action)
        
        # Save
        save_icon = QIcon.fromTheme("document-save")
        save_action = QAction(save_icon, "Save", self)
        save_action.setStatusTip("Save File")
        save_action.setToolTip("")
        save_action.triggered.connect(self.save_file)
        tool_bar.addAction(save_action)
        
        # Toggle view
        toggle_view_action = QAction("Toggle View", self)
        toggle_view_action.setStatusTip("Toggle between minimal and maximal")
        toggle_view_action.setToolTip("")
        toggle_view_action.triggered.connect(self.toggle_view)
        tool_bar.addAction(toggle_view_action)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open .npy File", "", "NumPy Files (*.npy)"
        )
        if file_path:
            current_view = self.stacked_widget.currentWidget()
            # Pass to drop label method
            if hasattr(current_view, 'drop_label'):
                current_view.drop_label.load_file(file_path)
            else:
                # If minimal or maximal view is structured differently
                pass

    def save_file(self):
        """Save the current matplotlib figure to a file."""
        current_view = self.stacked_widget.currentWidget()
        if not hasattr(current_view, 'canvas') or not current_view.canvas.axes.collections:
            QMessageBox.warning(self, "No Data", "No plot available to save.")
            return
            
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                # Get figure from current canvas
                figure = current_view.canvas.figure
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Save with tight layout and high DPI
                figure.savefig(
                    file_path,
                    bbox_inches='tight',
                    dpi=300,
                    format=os.path.splitext(file_path)[1][1:]
                )
                self.status.showMessage(f"Plot saved to {file_path}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot: {str(e)}")

    def toggle_view(self):
        current = self.stacked_widget.currentWidget()
        if current == self.minimal_view:
            print("Switching to maximal view")
            self.maximal_view.show()
            self.stacked_widget.setCurrentWidget(self.maximal_view)
        else:
            print("Switching to minimal view")
            self.minimal_view.show()
            self.stacked_widget.setCurrentWidget(self.minimal_view)
                
    def open_umap_settings(self):
        dialog = UMAPSettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.umap_settings = {
                "n_neighbors": dialog.n_neighbors,
                "min_dist": dialog.min_dist,
                "n_components": dialog.n_components,
                "metric": dialog.metric
            }
            self.status.showMessage("UMAP settings updated.", 5000)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyleSheet("QToolTip { opacity: 0; }")  # Make tooltips invisible
    
    set_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
