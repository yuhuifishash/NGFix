# train_number=10 #10M
# test_number=10 #10K
# Kh=100
# Nq=100
# MEX=48
# K=100
# taskset -c 1 ./search_ex_hnsw --base_data_path /SSD/Text-to-Image/base.10M.fbin \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --base_graph_path /SSD/models/hnsw/hnsw_t2i_m16_ef2000_ip \
# --test_number 10 \
# --metric ip_float --K ${K} --result_path /home/hzy/EXlib/result/t2i10M_ngfix_K${K}.csv \
# --index_path /SSD/models/EX/Text2Image/t2i10M_EXHNSW_Kh${Kh}_M${MEX}_Nq${Nq}_Train${train_number}M.index \


# taskset -c 1 search_hnsw_ngfix --base_data_path /SSD/Text-to-Image/base.10M.fbin \
# --test_query_path /SSD/Text-to-Image/query.10k.fbin \
# --test_gt_path /SSD/Text-to-Image/gt.10K_10M.bin \
# --test_number 10 \
# --metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_t2i.csv \
# --index_path /SSD/models/hnsw/test_t2i \

taskset -c 1 search_hnsw_ngfix --base_data_path /SSD/WebVid/webvid.base.2.5M.fbin \
--test_query_path /SSD/WebVid/webvid.query.10k.fbin \
--test_gt_path /SSD/WebVid/gt.query.top100.bin \
--test_number 10 \
--metric ip_float --K 100 --result_path /home/hzy/NGFix2/result/test_web.csv \
--index_path /SSD/models/hnsw/hnsw_ngfix_webvid \