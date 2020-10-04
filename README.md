# splatoon_competition

===
- [対戦ゲームデータ分析甲子園](https://prob.space/competitions/game_winner)

# Structures
```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── train_data.csv(実行前に配置)
│   │   └── test_data.csv(実行前に配置)
│   │   └── statink-weapon2.csv(実行前に配置)
│   │   └── stagedata.csv(実行前に配置)
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models
│   └── lgbm.py
├── notebooks
│   ├── 01.eda.ipynb
│   └── 02.check_enginned_mart.ipynb
├── scripts
│   └── convert_to_feather.py
├── utils
│   └── __init__.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py
└── tox.ini
```
# Commands

## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## Run LightGBM

```
python run.py
```

## flake8

```
flake8 .
```


# memo

## try1

### RUN
```
python scripts/convert_to_feather.py
python features/create.py
python run.py
```

### SUBMIT
```
sub_20200922132856_0.6846010969520098.csv
```

### CV
[0.6846087941610486, 0.6832817651930335, 0.6837258456833603, 0.6854290826037651, 0.6828396269027153, 0.6843795537970178, 0.6860389023055714, 0.6848122665924761, 0.6873121458304
686, 0.6835829864506402]

0.6846010969520098

### Public score
0.549330

### comment
object型を対象にtarget encodingにより特徴量を作成。


## try2

### RUN
```
python scripts/convert_to_feather.py
python features/create_v01.py
python run_v01.py
```

### SUBMIT
```
sub_20201004004037_0.682251709808842.csv
```

### CV
[0.6820526394530785, 0.6801965026164987, 0.6828405384955627, 0.6825201262737013, 0.6809111269589126, 0.6813751804065138, 0.6835319799184765, 0.6829363079521379, 0.6832609981300
778, 0.6828916978834589]

0.682251709808842

### Public score
0.551023	

### comment
特徴量に以下２つを追加
①A、Bチーム内のcharger数のカウント
②A、Bチーム内のランク、レベルの平均値、偏差値、最小値、最大値


## try3

### RUN
```
python scripts/convert_to_feather.py
python features/create_v02.py
python run_v02.py
```

### SUBMIT
```
sub_20201004110310_0.6820503068482273.csv
```

### CV
[0.6817056047099685, 0.6790334185525244, 0.6834953466510034, 0.6823131165732748, 0.6790714584691785, 0.681658339089119, 0.6833403400830539, 0.6826736895256236, 0.68426178322304
25, 0.6829499716054859]

0.6820503068482273

### Public score
0.552152

### comment
特徴量に以下２つを追加
①A、Bチーム内の対象スペシャル技（ナイス玉、ウルトラハンコ、アーマー）のカウント
②ステージエリア
