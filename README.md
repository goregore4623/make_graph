# matplotlibでグラフを作成するプログラム

## environment
python 3.9.10 \

## Installation
```bash
pyenv install 3.9.10
```
```bash
pyenv virtualenv 3.9.10 make_graph
```
```bash
pyenv local make_graph
```
```bash
pip install -r requirements.txt
```

### make graph

```bash
python make_bar_graph.py \
-d ${HOME}/make_graph/data/bar_data.csv \
-o ${HOME}/ffmpeg-visualize/results \
-p bar_graph
```


results are in `results/bar_graph.pdf`


### Optional
