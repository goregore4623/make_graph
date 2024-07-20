import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import japanize_matplotlib
from pathlib import Path
import argparse
class Scatter_Graph_Plotter:
    """
    折れ線グラフを作成するクラス
        data: データセット
        xylabel: X軸とY軸のラベル
        y_lim: Y軸の範囲
        figsize: グラフのサイズ
        font_size: フォントサイズ
        scale: グラフの拡大率
    """
    def __init__(self, 
                 data, 
                 xylabel,  
                 y_lim=[0.0, 1.0], 
                 figsize=[3.1, 1.8], 
                 font_size=8, 
                 scale=1.0):
        self.data = data
        self.xlabel, self.ylabel = xylabel
        self.y_lim = y_lim
        self.figsize = figsize
        self.font_size = font_size * scale
        self.scale = scale
        self.fig, self.ax = plt.subplots(figsize=(figsize[0] * scale, figsize[1] * scale), layout="constrained")
    
    def setup_graph(self):
        # グリッドと軸の設定
        self.ax.grid(which="major", axis="y", color="gray", linestyle="dashed", linewidth=0.1 * self.scale)
        self.ax.yaxis.set_major_locator(MultipleLocator(0.2)) # Y軸の補助線を0.2刻みにする
        self.ax.set_axisbelow(True)
        self.ax.set_ylim(self.y_lim[0], self.y_lim[1])

        # 軸の枠線の設定
        for spine in self.ax.spines.values():
            spine.set_linewidth(0.5 * self.scale)  # 太さを適宜調整

        # 上と右の軸を非表示にする
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # 軸メモリの設定
        self.ax.get_xaxis().set_tick_params(pad=2 * self.scale)
        self.ax.tick_params(axis='both', which='major', labelsize=self.font_size * 0.8)  # メモリのフォントサイズを設定
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # X軸のメモリを整数にする
    
    def plot_data(self):
        # データのプロット
        y_index = self.data.columns[0]
        legend_labels = self.data.columns[1:]
        for legend_label, x_data in self.data.iloc[:, 1:].items():
            self.ax.plot(self.data[y_index], x_data, label=legend_label)
        
        # x軸を逆順にする（角度が大きい方から小さい方へ）
        plt.gca().invert_xaxis()
        
        # 軸ラベルの設定
        self.ax.set_xlabel(self.xlabel, fontsize=self.font_size, labelpad=2 * self.scale)
        self.ax.set_ylabel(self.ylabel, fontsize=self.font_size, labelpad=2 * self.scale)
    
    def setup_legend(self):
        # 凡例の設定
        legend_labels = self.data.columns[1:]
        self.ax.legend(legend_labels, loc='upper center', fontsize=self.font_size * 0.8, bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    
    def show(self):
        plt.show()
        
    def save(self, output_file):
        plt.savefig(output_file)
        print(f"[INFO] Save the graph as {output_file}")

def main(args):
    # タブ区切りファイルを読み込む
    data = pd.read_csv(args.data_file, sep='\t')
    
    plotter = Scatter_Graph_Plotter(
    data=data,
    xylabel=args.xylabel,
    y_lim=[0.0, 1],
    figsize=args.figsize,
    font_size=args.font_size,
    scale=args.scale
    )
    
    # グラフの設定
    plotter.setup_graph()
    
    # データのプロット
    plotter.plot_data()
    plotter.setup_legend()

    # グラフの保存
    output_file = Path(args.output_dir).joinpath(f"{args.prefix}.pdf") # 保存する画像のファイル名
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
        '-s','--scale', 
        type=float, 
        default=1.0,
        help='scale of graph default:for paper, 3.0:for presentation')
    parser.add_argument(
        '--xylabel',
        nargs=2, 
        default=['Angle of View','Accuracy'], 
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

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
