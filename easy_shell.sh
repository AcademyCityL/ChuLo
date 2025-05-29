# python run_pl_ex.py --config ./config/imdb_vanilla.toml && \
# python run_pl_ex.py --config ./config/hp_vanilla.toml && \
# python run_pl_ex.py --config ./config/mr_key_phrase_chunk_rep.toml && \
# python run_pl_ex.py --config ./config/lun_key_phrase_chunk_rep.toml
# python run_pl_ex.py --config ./config/r8_vanilla.toml && \
# python run_pl_ex.py --config ./config/bbcn_vanilla.toml && \
# python run_pl_ex.py --config ./config/r8_key_phrase_chunk_rep.toml && \
# python run_pl_ex.py --config ./config/bbcn_key_phrase_chunk_rep.toml
# python run_pl_ex.py --config ./config/lun_key_phrase_chunk_rep.toml && \
# python run_pl_ex.py --config ./config/lun_vanilla.toml && \
# python run_pl_ex.py --config ./config/r8_key_phrase_chunk_rep.toml && \
# python run_pl_ex.py --config ./config/bbcn_key_phrase_chunk_rep.toml && \
# python run_pl_ex.py --config ./config/mr_key_phrase_chunk_rep.toml
# python sentence_rank.py --config ./config/lun_key_phrase_split_all_locs.toml && \
# python run_pl_ex.py --config ./config/lun_key_phrase_chunk_rep_bert.toml && \
# python run_pl_ex.py --config ./config/lun_key_phrase_chunk_rep_sep_sent_imp_bert.toml && \

# python sentence_rank.py --config ./config/bs-pair_key_phrase_split_all_locs.toml && \
python run_pl_ex.py --config ./config/bs-pair_key_phrase_chunk_rep_bert_elu_sum2.toml 
# # python run_pl_ex.py --config ./config/bs-pair_key_phrase_chunk_rep_bert_sep_sent_imp.toml && \
# python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert.toml
# python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert.toml && \
# python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert.toml
# python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert.toml
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 0 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 1 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 2 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 3 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 4 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 5 --split_size 1000 --whole_doc ;
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 6 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 7 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 8 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 9 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 10 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 11 --split_size 1000 --whole_doc ;
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 12 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 13 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 14 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 15 --split_size 1000 --whole_doc &
# python key_phrase_split_analysis.py --config ./config/bs_key_phrase_split_all_locs.toml --split_idx 16 --split_size 1000 --whole_doc
# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert.toml > ./log/bsf1111.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert2.toml > ./log/bsf2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert3.toml > ./log/bsf3.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert.toml > ./log/bsf1.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert-1.toml > ./log/bsf2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert-2.toml > ./log/bsf3.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert-2-3.toml > ./log/bsf3-3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert-2-2.toml > ./log/bsf3-2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/bs_key_phrase_chunk_rep_bert-2.toml > ./log/bsf3.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/bs-pair_key_phrase_chunk_rep_bert.toml > ./log/bspf1.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/bs-pair_key_phrase_chunk_rep_bert2.toml > ./log/bspf2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/bs-pair_key_phrase_chunk_rep_bert3.toml > ./log/bspf3.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/ng20_key_phrase_chunk_rep_bert.toml > ./log/ngf111.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/ng20_key_phrase_chunk_rep_bert2.toml > ./log/ngf222.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/ng20_key_phrase_chunk_rep_bert3.toml > ./log/ngf333.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert.toml > ./log/euf1.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert1.toml > ./log/euf2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert2.toml > ./log/euf3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert3.toml > ./log/euf4.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert4.toml > ./log/euf5.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pl_ex.py --config ./config/eurlex_key_phrase_chunk_rep_bert5.toml > ./log/euf6.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert.toml > ./log/euif1.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert1.toml > ./log/euif2.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert2.toml > ./log/euif3.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert3.toml > ./log/euif4.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert4.toml > ./log/euif5.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert5.toml > ./log/euif6.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python run_pl_ex.py --config ./config/eurlex-inverse_key_phrase_chunk_rep_bert7.toml > ./log/euif7.txt 2>&1 &

# python pre_tokenize.py --config ./config/bs-pair_key_phrase_split_all_locs-vanillat.toml;
# python pre_tokenize.py --config ./config/bs-pair_key_phrase_split_all_locs.toml;
# python pre_tokenize.py --config ./config/bs-pair_key_phrase_split2_all_locs-vanillat.toml;
# python pre_tokenize.py --config ./config/bs-pair_key_phrase_split2_all_locs.toml;
# python pre_tokenize.py --config ./config/eurlex-inverse_key_phrase_split_all_locs-vanillat.toml;
# python pre_tokenize.py --config ./config/eurlex-inverse_key_phrase_split_all_locs.toml;
# python pre_tokenize.py --config ./config/eurlex-inverse_key_phrase_split_all_locs-vanillat.toml;
# python pre_tokenize.py --config ./config/eurlex-inverse_key_phrase_split_all_locs.toml;
# python pre_tokenize.py --config ./config/eurlex_key_phrase_split_all_locs-vanillat.toml;
# python pre_tokenize.py --config ./config/eurlex_key_phrase_split_all_locs.toml;
# python pre_tokenize.py --config ./config/eurlex_key_phrase_split_all_locs-vanillat.toml;
# python pre_tokenize.py --config ./config/eurlex_key_phrase_split_all_locs.toml;
# python key_phrase_split_analysis.py --config ./config/eurlex_key_phrase_split_all_locs.toml --whole_doc --not_split && \
# python key_phrase_split_analysis.py --config ./config/eurlex_key_phrase_split2_all_locs.toml --whole_doc --not_split
# python pre_tokenize.py --config ./config/eurlex_key_phrase_split_all_locs.toml && \
# python pre_tokenize.py --config ./config/eurlex_key_phrase_split2_all_locs.toml