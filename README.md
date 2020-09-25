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

## take1

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
