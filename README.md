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
```
[0.6846087941610486, 0.6832817651930335, 0.6837258456833603, 0.6854290826037651, 0.6828396269027153, 0.6843795537970178, 0.6860389023055714, 0.6848122665924761, 0.6873121458304
686, 0.6835829864506402]

0.6846010969520098
```

### Public score
```
0.549330
```

### comment
```
object型を対象にtarget encodingにより特徴量を作成。
```

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
```
[0.6820526394530785, 0.6801965026164987, 0.6828405384955627, 0.6825201262737013, 0.6809111269589126, 0.6813751804065138, 0.6835319799184765, 0.6829363079521379, 0.6832609981300
778, 0.6828916978834589]

0.682251709808842
```
### Public score
```
0.551023	
```

### comment
```
特徴量に以下２つを追加
①A、Bチーム内のcharger数のカウント
②A、Bチーム内のランク、レベルの平均値、偏差値、最小値、最大値
```


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
```
[0.6817056047099685, 0.6790334185525244, 0.6834953466510034, 0.6823131165732748, 0.6790714584691785, 0.681658339089119, 0.6833403400830539, 0.6826736895256236, 0.68426178322304
25, 0.6829499716054859]

0.6820503068482273
```

### Public score
```
0.552152
```

### comment
```
特徴量に以下２つを追加
①A、Bチーム内の対象スペシャル技（ナイス玉、ウルトラハンコ、アーマー）のカウント
②ステージエリア
```


## try4

### RUN
```
python scripts/convert_to_feather.py
python features/create_v03.py
python run_v03.py
```

### SUBMIT
```
sub_20201004121824_0.6814406466368689.csv
```

### CV
```
[0.6814191283178093, 0.6778783209771855, 0.6823654923616428, 0.6823080957089523, 0.6805800575586959, 0.6800786470499058, 0.6820083582459349, 0.6823153440680323, 0.6836033036266
121, 0.6818497184539164]
```

### Public score
```
0.553705
```

### comment
```
特徴量を改修
①A、Bチーム内のすべてのスペシャル技を対象にカウント数を集計
②A、Bチーム内のすべてのブキカテゴリーを対象にカウント数を集計
```


## try5

### RUN
```
python scripts/convert_to_feather.py
python features/create_v04.py
python run_v04.py
```

### SUBMIT
```
sub_20201004132139_0.6816136596036713.csv
```

### CV
```
[0.6805428188754533, 0.6786951608236029, 0.6823828470805477, 0.6825471219029239, 0.6798071898825546, 0.6807888732008852, 0.6822896545872371, 0.6825351845064088, 0.6837366944295
076, 0.6828110507475921]
0.6816136596036713
```

### Public score
```
0.552858
```

### comment
```
特徴量を改修
①A、Bチーム内のすべてのサブウエポンを対象にカウント数を集計
```


## try6

### RUN
```
python scripts/convert_to_feather.py
python features/create_v06.py
python run_v06.py
```

### SUBMIT
```
sub_20201005224952_0.6806427546231666.csv
```

### CV
```
[0.6809555504384708, 0.6782471915297275, 0.681438509324889, 0.6817393285805751, 0.677491279
8996775, 0.6793277092244226, 0.6813084710362931, 0.6813723827876546, 0.6824212906700384, 0.
6821258327399171]
0.6806427546231666
```

### Public score
```
0.556669
```

### comment
```
特徴量を改修
①A、Bチーム内のレベル・ランクの平均値を差分に集約
②charger_flag等が重複していたので修正
```
