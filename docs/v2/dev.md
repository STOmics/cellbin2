mpp effective range: 1.67μm/pixel ~ 0.167μm/pixel

test
```shell

pip install pytest-html
python3 -m pytest # This is necessary, otherwise the cellbin will not be found
pytest test/test_cellbin_pipeline.py --html=/media/Data1/user/dengzhonghan/data/cellbin2/auto_test/0.0.1/report/0.01.html --self-contained-html
```

deploy
```shell
pip install git+https://github.com/STOmics/cellbin2.git@main
python -c "from cellbin2.utils.weights_manager import download_all_weights; download_all_weights()"
```

