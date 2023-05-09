set -e
mkdir -p log
python demo_deepsurv_tabular.py config_tabular_hypersphere.ini    > log/demo_deepsurv_tabular_hypersphere.txt
python demo_deepsurv_tabular.py config_tabular_no_hypersphere.ini > log/demo_deepsurv_tabular_no_hypersphere.txt
