import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')  # or 'Qt5Agg', 'Agg', etc.
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import umap


class MplCanvas(FigureCanvas):
    """
    Matplotlib Canvas Widget to display the scatter plot.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot_umap(self, points_2d):
        """
        Given a 2D array of points (N, 2), plot them in a scatter plot.
        """
        self.axes.clear()
        self.axes.scatter(points_2d[:, 0], points_2d[:, 1], s=10, alpha=0.7)
        self.axes.set_title("UMAP")
        self.axes.set_xlabel("Dimension 1")
        self.axes.set_ylabel("Dimension 2")
        self.draw()


class UMAPCallback(QObject):
    """Custom callback to handle UMAP progress updates"""
    progress_updated = Signal(int)
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.current = 0
        
    def __call__(self, *args):
        self.current += 1
        progress = int((self.current / self.total_epochs) * 100)
        self.progress_updated.emit(progress)


class DropLabel(QLabel):
    """
    A label that accepts .npy file drops, loads them,
    displays summary stats, and triggers a UMAP plot update.
    """
    def __init__(self, canvas, parent=None):
        super().__init__("Drag & Drop a .npy file here", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { border: 2px dashed #aaa; }")
        self.setAcceptDrops(True)
        self.canvas = canvas

        # Add progress bar (hidden by default)
        self.progress = QProgressBar(self)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.hide()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        
        # Add status label (hidden by default)
        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.hide()
        
        # Layout for progress indicators
        self.layout = QVBoxLayout(self)
        self.layout.addStretch()
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress)
        self.layout.addStretch()

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
                    # Show loading UI
                    self.setText("")
                    self.status_label.setText("Loading data...")
                    self.status_label.show()
                    self.progress.setRange(0, 0)  # Infinite progress bar
                    self.progress.show()
                    
                    data = np.load(file_path)
                    
                    if data.ndim == 2 and data.shape[0] > 1:
                        self.status_label.setText("Computing UMAP embedding...")
                        # Use QTimer to allow UI to update before heavy computation
                        QTimer.singleShot(100, lambda: self.compute_umap(data))
                    else:
                        self.show_error("Data shape not suitable for UMAP")
                        
    def compute_umap(self, data):
        try:
            # Create callback for progress updates
            n_epochs = 200  # Default UMAP epochs
            callback = UMAPCallback(n_epochs)
            callback.progress_updated.connect(self.update_progress)
            
            umap_reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42,
                verbose=True,  # Enable verbose output
                n_epochs=n_epochs  # Explicitly set epochs
            )
            
            self.status_label.setText("Computing UMAP embedding...")
            self.progress.setValue(0)
            embedding_2d = umap_reducer.fit_transform(data, callbacks=callback)
            
            self.canvas.plot_umap(embedding_2d)
            
            # Show data summary
            summary_lines = [
                f"Shape: {data.shape}",
                f"Data Type: {data.dtype}",
                f"Min: {data.min():.4f}",
                f"Max: {data.max():.4f}",
                f"Mean: {data.mean():.4f}",
                f"Std Dev: {data.std():.4f}",
            ]
            summary_text = "\n".join(summary_lines)
            self.setText(summary_text)
        except Exception as e:
            self.show_error(f"Error computing UMAP: {str(e)}")
        finally:
            self.progress.hide()
            self.status_label.hide()

    def update_progress(self, value):
        """Update progress bar value"""
        self.progress.setValue(value)
            
    def show_error(self, message):
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
        self.status_label.hide()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Embeddings UMAP Visualizer")
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.drop_label = DropLabel(canvas=self.canvas)
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.drop_label)
        main_layout.addWidget(self.canvas)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.resize(800, 600)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
