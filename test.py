import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QTabWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QLabel, QComboBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.arima.model import ARIMA

class AppIA(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plateforme IA - Hind - Dr. EL MKHALET MOUNA")
        self.setGeometry(50, 50, 1300, 850)
        
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
      
        self.tab_reg = QWidget(); self.tab_clust = QWidget()
        self.tab_rf = QWidget(); self.tab_ts = QWidget()
        self.tab_nn = QWidget(); self.tab_cv = QWidget()
        
     
        self.setup_regression_tab(); self.setup_clustering_tab()
        self.setup_rf_tab(); self.setup_ts_tab()
        self.setup_nn_tab(); self.setup_cv_tab()
        
        self.tabs.addTab(self.tab_reg, "Régression")
        self.tabs.addTab(self.tab_clust, "Clustering")
        self.tabs.addTab(self.tab_rf, "Random Forest")
        self.tabs.addTab(self.tab_ts, "Time Series")
        self.tabs.addTab(self.tab_nn, "Réseaux Neurones")
        self.tabs.addTab(self.tab_cv, "Validation Croisée")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    
    def setup_regression_tab(self):
        l = QHBoxLayout(); p = QVBoxLayout(); f = QFormLayout()
        self.c_reg = QComboBox(); self.c_reg.addItems(["Simple (2D)", "Multiple (3D)"])
        self.i_min = QLineEdit("-10"); self.i_max = QLineEdit("10")
        self.b_reg = QPushButton("Lancer"); self.lab_reg = QLabel("Res: --")
        f.addRow("Type:", self.c_reg); f.addRow("Min X:", self.i_min); f.addRow("Max X:", self.i_max)
        p.addLayout(f); p.addWidget(self.b_reg); p.addWidget(self.lab_reg); p.addStretch()
        self.fig_reg = plt.figure(); self.can_reg = FigureCanvas(self.fig_reg)
        l.addLayout(p, 1); l.addWidget(self.can_reg, 3); self.tab_reg.setLayout(l)
        self.b_reg.clicked.connect(self.do_reg)

    def setup_clustering_tab(self):
        l = QHBoxLayout(); p = QVBoxLayout(); f = QFormLayout()
        self.i_k = QLineEdit("3"); self.b_cl = QPushButton("Clustering"); self.lab_cl = QLabel("Res: --")
        f.addRow("K:", self.i_k); p.addLayout(f); p.addWidget(self.b_cl); p.addWidget(self.lab_cl); p.addStretch()
        self.fig_cl = plt.figure(); self.can_cl = FigureCanvas(self.fig_cl)
        l.addLayout(p, 1); l.addWidget(self.can_cl, 3); self.tab_clust.setLayout(l)
        self.b_cl.clicked.connect(self.do_cl)

    def setup_rf_tab(self):
        l = QHBoxLayout(); p = QVBoxLayout(); f = QFormLayout()
        self.i_rf = QLineEdit("100"); self.b_rf = QPushButton("Random Forest"); self.lab_rf = QLabel("Acc: --")
        f.addRow("Arbres:", self.i_rf); p.addLayout(f); p.addWidget(self.b_rf); p.addWidget(self.lab_rf); p.addStretch()
        self.fig_rf = plt.figure(); self.can_rf = FigureCanvas(self.fig_rf)
        l.addLayout(p, 1); l.addWidget(self.can_rf, 3); self.tab_rf.setLayout(l)
        self.b_rf.clicked.connect(self.do_rf)

    def setup_ts_tab(self):
        l = QHBoxLayout(); p = QVBoxLayout(); self.b_ts = QPushButton("ARIMA(1,1,1)")
        p.addWidget(self.b_ts); p.addStretch(); self.fig_ts = plt.figure(); self.can_ts = FigureCanvas(self.fig_ts)
        l.addLayout(p, 1); l.addWidget(self.can_ts, 3); self.tab_ts.setLayout(l)
        self.b_ts.clicked.connect(self.do_ts)

    
    def setup_nn_tab(self):
        l = QHBoxLayout(); p = QVBoxLayout(); f = QFormLayout()
        self.i_nn = QLineEdit("10,10") 
        self.b_nn = QPushButton("Lancer Neurones")
        self.lab_nn = QLabel("Erreur: --")
        f.addRow("Couches (ex: 10,10):", self.i_nn)
        p.addLayout(f); p.addWidget(self.b_nn); p.addWidget(self.lab_nn); p.addStretch()
        self.fig_nn = plt.figure(); self.can_nn = FigureCanvas(self.fig_nn)
        l.addLayout(p, 1); l.addWidget(self.can_nn, 3); self.tab_nn.setLayout(l)
        self.b_nn.clicked.connect(self.do_nn)

    
    def setup_cv_tab(self):
        l = QHBoxLayout(); p = QVBoxLayout()
        self.b_cv = QPushButton("Comparer les Modèles")
        p.addWidget(self.b_cv); p.addStretch()
        self.fig_cv = plt.figure(); self.can_cv = FigureCanvas(self.fig_cv)
        l.addLayout(p, 1); l.addWidget(self.can_cv, 3); self.tab_cv.setLayout(l)
        self.b_cv.clicked.connect(self.do_cv)

   
    def do_reg(self):
        tipo = self.c_reg.currentText(); x_min, x_max = float(self.i_min.text()), float(self.i_max.text()); self.fig_reg.clear()
        if "Simple" in tipo:
            X = np.random.uniform(x_min, x_max, 50).reshape(-1, 1); Y = 2.5 * X + 10 + np.random.normal(0, 2, X.shape)
            m = LinearRegression().fit(X, Y); ax = self.fig_reg.add_subplot(111); ax.scatter(X, Y, color='blue'); ax.plot(X, m.predict(X), color='red')
            self.lab_reg.setText(f"a={m.coef_[0][0]:.2f}, b={m.intercept_[0]:.2f}")
        else:
            X1, X2 = np.random.uniform(x_min, x_max, 100), np.random.uniform(x_min, x_max, 100); X = np.column_stack((X1, X2)); Y = 3*X1 - 2*X2 + 5 + np.random.normal(0, 2, 100)
            m = LinearRegression().fit(X, Y); ax = self.fig_reg.add_subplot(111, projection='3d'); ax.scatter(X1, X2, Y, c='red')
        self.can_reg.draw()

    def do_cl(self):
        k = int(self.i_k.text()); data = np.random.randn(200, 2); km = KMeans(n_clusters=k).fit(data)
        self.fig_cl.clear(); ax = self.fig_cl.add_subplot(111); ax.scatter(data[:, 0], data[:, 1], c=km.labels_); ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red', marker='*', s=200)
        self.can_cl.draw()

    def do_rf(self):
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=200, centers=3); clf = RandomForestClassifier(n_estimators=int(self.i_rf.text())).fit(X, y)
        self.fig_rf.clear(); ax = self.fig_rf.add_subplot(111); ax.bar(['X1', 'X2'], clf.feature_importances_); self.can_rf.draw()

    def do_ts(self):
        h = np.cumsum(np.random.normal(0.5, 1, 80)); m = ARIMA(h, order=(1,1,1)).fit(); f = m.forecast(steps=15)
        self.fig_ts.clear(); ax = self.fig_ts.add_subplot(111); ax.plot(h, label='Madhi'); ax.plot(np.arange(80,95), f, label='Future'); ax.legend(); self.can_ts.draw()

    def do_nn(self):
       
        X = np.linspace(0, 10, 100).reshape(-1, 1); Y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)
        layers = tuple(map(int, self.i_nn.text().split(',')))
        nn = MLPRegressor(hidden_layer_sizes=layers, max_iter=1000).fit(X, Y)
        self.fig_nn.clear(); ax = self.fig_nn.add_subplot(111); ax.scatter(X, Y, color='gray'); ax.plot(X, nn.predict(X), color='purple', label='NN Prediction')
        ax.legend(); self.can_nn.draw(); self.lab_nn.setText(f"NN kheddam b architecture {layers}")

    def do_cv(self):
       
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        models = [RandomForestClassifier(), MLPClassifier(max_iter=500)]
        names = ['Random Forest', 'Réseau Neurones']
        scores = [cross_val_score(m, X, y, cv=5).mean() for m in models]
        
        self.fig_cv.clear(); ax = self.fig_cv.add_subplot(111)
        ax.bar(names, scores, color=['blue', 'orange'])
        ax.set_ylim(0.8, 1.0); ax.set_title("Comparaison des Exactitudes (CV=5)")
        self.can_cv.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppIA()
    window.show()
    sys.exit(app.exec_())


