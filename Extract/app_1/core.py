import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import pandas as pd
import pickle
from scipy.spatial import procrustes

class DataManager:
    def __init__(self, dir_path):
        
        self.dir_path = dir_path
        self.directories = self.get_directories(self.dir_path)
        ## Todo 
        # choose file name
        self.file_names = {}
        self.data = Data(self.load_df(os.path.join(self.dir_path + self.directories)), self.load_embeddings(os.path.join(self.dir_path + self.directories)))
    def get_directories(dir_path):
        return [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    def load_df(file_path):
        return pd.read_csv(file_path, index_col=0)

    def load_embeddings(file_path):
        with open(file_path, "rb") as f:
            embeddings = np.array(pickle.load(f))
         
        return embeddings

    def preprocess(self) -> np.ndarray:
        
        return self.data

class Data:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings

class TransitionData:
    def __init__(self, data):
        self.from_data = data
        self.to_data = data
    def next(self, to_data):
        self.from_data = self.to_data
        self.to_data = to_data

class AnimationManager:
    pass  


class DimensionalityReducer:
    def __init__(self, method: str = "PCA"):
        self.methods = {
            "PCA": self._pca,
            "t-SNE": self._tsne
        }
        self.selected_method = method
    def get_methods(self):
        return self.methods.keys()

    def reduce(self, data: np.ndarray) -> np.ndarray:
        if self.method:
            dr = self.methods.get(self.selected_method, self._pca)
            return dr(data)
        raise ValueError(f"Unknown method:")
    ## Todo
    # PCAの行列を取得するなどできるようにしたい
    # ハイパーパラメータの調整の機能
    def _pca(self, data: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=2)
        return pca.fit_transform(data)

    def _tsne(self, data: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=2, random_state=42)
        return tsne.fit_transform(data)


class AlignmentHandler:
    """
    それぞれ次元削減した結果の data1, data2に関して、data2の結果をdata1にアライメントする
    """
    def __init__(self, method: str = "Procrustes"):
        self.methods = {
            "Procrustes": self._procrustes
        }
        self.method = method
    
    def _procrustes(self, data1:np.ndarray, data2: np.ndarray):
        _, aligned_data2, d = procrustes(data1, data2)

        scale_factor = np.std(data1) / np.std(data2)
        print(f"scale: {scale_factor:4f}")
        aligned_data2 *= scale_factor
        return 


    def align(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        # 仮の実装: メソッドに応じて異なるアライメントを実行可能
        if self.method:
            current_method = self.methods.get(self.selected_method, self._pca)
            return current_method(before, after)
        raise ValueError(f"Unknown alignment method: ")


class AnimationManager:

    def __init__(self, data_manager: DataManager ,alignment_handler: AlignmentHandler, dimensionality_reducer: DimensionalityReducer, transition_data: TransitionData):

        self.alignment_handler = alignment_handler
        self.dimensionality_reducer = dimensionality_reducer
        self.transition_data = transition_data
        self.data_manager = data_manager

    ## Todo
    # animationの種類への対応: scroll, squere_area, auto

    def initialize_figure(self):
        pass

    def generate_fig_plotly(self):
        fig = go.Figure()
        pass

    def draw_line(self, fig):
        pass




    def create_frames(self, before: np.ndarray, after: np.ndarray, steps: int = 20) -> list:
        frames = []
        
        return frames

class AnnotationManager:
    def __init__(self, fig):
        self.fig = fig

    def add_text():
