MEX=48
M=16
efC=500
taskset -c 1 search_hnsw_ngfix --base_data_path /SSD/Text-to-Image/base.10M.fbin \
--test_query_path /SSD/Text-to-Image/query.10k.fbin \
--test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
--metric ip_float --K 10 --result_path /home/hzy/NGFix2/result/test_t2i_K10.csv \
--index_path /SSD/models/NGFix/t2i10M_HNSW_NGFix_M${M}_efC${efC}_MEX${MEX}.index \


# taskset -c 1 search_hnsw_ngfix --base_data_path /SSD/WebVid/webvid.base.2.5M.fbin \
# --test_query_path /SSD/WebVid/webvid.query.10k.fbin \
# --test_gt_path /SSD/WebVid/gt.query.top100.bin \
# --test_number 10 \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_web.csv \
# --index_path /SSD/models/hnsw/hnsw_ngfix_webvid \


# taskset -c 1 search_hnsw_ngfix --base_data_path /SSD/Text-to-Image/base.10M.fbin \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --test_number 10 \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i.csv \
# --index_path /SSD/models/hnsw/test_t2i \

# gdb --args search_hnsw_ngfix --base_data_path /SSD/WebVid/webvid.base.2.5M.fbin \
# --test_query_path /SSD/WebVid/webvid.query.10k.fbin \
# --test_gt_path /SSD/WebVid/gt.query.top100.bin \
# --test_number 10 \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_web.csv \
# --index_path /SSD/models/hnsw/hnsw_ngfix_webvid \