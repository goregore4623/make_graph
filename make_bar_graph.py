import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path
import numpy as np
import argparse

class BarGraphPlotter:
    def __init__(self, 
                 data, 
                 xylabel,
                 figsize=[3.1, 1.8], 
                 font_size=8, 
                 scale=1.0,
                 show_datalabel=False,
                 bar_width=0.2, 
                 y_lim=[0.0, 1.0], ):
        self.data = data
        self.xylabel = xylabel
        self.figsize = figsize
        self.font_size = font_size
        self.scale = scale
        self.show_datalabel = show_datalabel
        self.bar_width = bar_width
        self.y_lim = y_lim
        
        self.fig, self.ax = None, None
        self.data_labels, self.methods, self.method_vals = None, None, None
        self._prepare_data() # データの準備

    def _prepare_data(self):
        """
        データのロード
        ラベルとメソッドを取得
        """
        self.data_labels = self.data.iloc[:, 0].values.tolist()
        self.methods = self.data.columns[1:].tolist()
        self.method_vals = [self.data[col].tolist() for col in self.data.iloc[:, 1:].columns]

    def setup_graph(self):
        """
        グラフの設定
        グリッド、軸、メモリ、枠線の設定
        """
        self.font_size *= self.scale
        self.fig, self.ax = plt.subplots(figsize=(self.figsize[0] * self.scale, self.figsize[1] * self.scale), layout="constrained")
        
        # グリッドの設定
        self.ax.grid(which="major", axis="y", color="gray", linestyle="dashed", linewidth=0.1 * self.scale) 
        self.ax.set_axisbelow(True) # グリッドを背面に表示
        
        # 軸の枠線を太くする
        for spine in self.ax.spines.values():
            spine.set_linewidth(0.5 * self.scale) # 太さを適宜調整
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        # Y軸の設定
        self.ax.set_ylim(self.y_lim[0], self.y_lim[1])
        self.ax.tick_params(direction='inout', length=3 * self.scale, width=0.5 * self.scale, colors='black', labelsize=self.font_size)
        
        # X軸の設定
        self.ax.set_xticks([pos + self.bar_width * (len(self.methods) - 1) / 2 for pos in range(len(self.data_labels))])
        self.ax.set_xticklabels(self.data_labels, fontsize=self.font_size * 0.6)
        self.ax.get_xaxis().set_tick_params(pad=2 * self.scale)
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="center") # X軸ラベルの角度を回転

    def plot_data(self):
        """
        データのプロット
        """
        bar_pos = range(len(self.data_labels))
        for data_label, method, method_val in zip(self.data_labels, self.methods, self.method_vals):
            # データをプロット
            self.ax.bar(bar_pos, method_val, width=self.bar_width, label=data_label)
            
            if self.show_datalabel: # データの数値を表示
                self.ax.bar_label(self.ax.containers[-1], fmt='%.3f',fontsize=self.font_size*0.4)
    
            bar_pos = [pos + self.bar_width for pos in bar_pos] # X軸の位置調整

    def setup_labels(self):
        """
        XY軸のラベルと凡例の設定
        """
        self.ax.set_xlabel(self.xylabel[0], fontsize=self.font_size, labelpad=2 * self.scale)
        self.ax.set_ylabel(self.xylabel[1], fontsize=self.font_size, labelpad=2 * self.scale)

    def setup_legend(self):
        """
        凡例の設定
        """
        self.ax.legend(self.methods, loc='upper center', fontsize=self.font_size * 0.8, bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
        
    def set_average(self):
        # 平均値を計算
        self.method_vals=np.array(self.method_vals)
        # 行ごとの平均値を計算
        row_means = np.mean(np.array(self.method_vals), axis=1)
        # 平均値を行列に追加
        self.method_vals = np.concatenate((self.method_vals, row_means[:, np.newaxis]), axis=1).tolist()
        
        # データセットのラベルを変更
        self.data_labels.append('Avg.')
        
    def show(self):
        """
        グラフの表示 jupyterのみ
        """
        plt.show()
        
    def save(self, output_file: Path):
        """
        グラフの保存
         - output_file: 保存するファイル名
        """
        plt.savefig(output_file)
        print(f"[INFO] Save the graph as {output_file}")
        

def main(args):
    # タブ区切りファイルを読み込む
    data = pd.read_csv(args.data_file, sep='\t')
    
    # グラフの設定
    plotter = BarGraphPlotter(data=data,
                 xylabel=args.xylabel, 
                 figsize=args.figsize,
                 font_size=args.font_size,
                 scale=args.scale,
                 show_datalabel=args.show_datalabel,
                 bar_width=0.2, 
                 y_lim=[0.0, 1.0],)
    
    # 平均値を計算
    if args.show_average:
        plotter.set_average()
    
    # グラフの作成
    plotter.setup_graph()
    plotter.plot_data()
    plotter.setup_labels()
    plotter.setup_legend()

    # グラフの保存
    output_file = Path(args.output_dir).joinpath(args.prefix+".pdf") # 保存する画像のファイル名
    plotter.save(output_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # file path
    parser.add_argument(
        "-d", "--data_file", 
        required=True, 
        help="path/to/output/dir")
    parser.add_argument(
        "-o", "--output_dir", 
        required=True, 
        help="path/to/output/dir")
    parser.add_argument(
        "-p", "--prefix", 
        default='Scatter_Graph', 
        help="result/file/name")
    
    # options
    parser.add_argument(
        '--xylabel',
        nargs=2, 
        default=['Camera pairs','Precision'], 
        help='x and y label')
    parser.add_argument(
        '--figsize', 
        nargs=2, type=float, 
        default=[3.1, 1.8], 
        help='figure size')
    parser.add_argument(
        '--font_size', 
        type=int, 
        default=8, 
        help='font size')
    parser.add_argument(
        '-s','--scale', 
        type=float, 
        default=1.0,
        help='scale of graph default:for paper, 3.0:for presentation')
    parser.add_argument(
        '-a','--show_average', 
        action='store_true', 
        default=False, 
        help='show average')
    parser.add_argument(
        '--show_datalabel', 
        action='store_true', 
        default=False, 
        help='add data label')

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
