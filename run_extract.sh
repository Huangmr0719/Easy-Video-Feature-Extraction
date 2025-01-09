#extract features
CUDA_VISIBLE_DEVICES=0 python visual_extract.py \
--csv /yout/preferred/csv/path.csv \
--model_path /your/model/path \
--extracted your_feature_name 

# merge features
python merge_features.py \
--folder your/features/file/path \
--output_path /your/feature/path.pth
