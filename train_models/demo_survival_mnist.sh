set -e
mkdir -p log
python demo_deepsurv_image.py config_image_hypersphere.ini > log/demo_deepsurv_image_hypersphere.txt
# python demo_deepsurv_image.py config_image_no_hypersphere.ini > log/demo_deepsurv_image_no_hypersphere.txt
